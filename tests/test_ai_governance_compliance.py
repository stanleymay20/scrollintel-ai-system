"""
AI Governance Compliance Tests

This module contains comprehensive tests for AI governance, ethics frameworks,
regulatory compliance, and public policy systems to ensure global regulatory compliance.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from scrollintel.engines.ai_governance_engine import AIGovernanceEngine
from scrollintel.engines.regulatory_compliance_engine import RegulatoryComplianceEngine, Regulation
from scrollintel.engines.ethical_decision_engine import EthicalDecisionEngine, EthicalPrinciple
from scrollintel.engines.public_policy_engine import PublicPolicyEngine, PolicyArea
from scrollintel.models.ai_governance_models import (
    AIGovernance, EthicsFramework, ComplianceStatus, RiskLevel
)


class TestAIGovernanceCompliance:
    """Test AI governance compliance capabilities"""
    
    @pytest.fixture
    def governance_engine(self):
        return AIGovernanceEngine()
    
    @pytest.fixture
    def sample_governance_config(self):
        return {
            "name": "Enterprise AI Governance Framework",
            "description": "Comprehensive AI governance for enterprise deployment",
            "policies": {
                "safety_requirements": {
                    "minimum_safety_score": 0.8,
                    "required_testing": ["robustness", "fairness", "privacy"],
                    "approval_threshold": 0.9
                },
                "ethics_requirements": {
                    "ethical_review_required": True,
                    "stakeholder_consultation": True,
                    "transparency_level": "high"
                }
            },
            "risk_thresholds": {
                "safety_score": 0.7,
                "compliance_score": 0.8,
                "ethical_score": 0.75
            }
        }
    
    @pytest.fixture
    def sample_ai_system_config(self):
        return {
            "system_id": "test_ai_system_001",
            "system_type": "machine_learning",
            "deployment_context": "production",
            "data_processing": True,
            "decision_making": True,
            "high_risk_application": True,
            # Safety features
            "adversarial_training": True,
            "input_validation": True,
            "error_handling": True,
            "testing_coverage": 0.85,
            "explainable_ai": True,
            "model_transparency": True,
            "decision_logging": True,
            "bias_testing": True,
            "demographic_parity": True,
            "equal_opportunity": True,
            "data_encryption": True,
            "differential_privacy": True,
            "data_minimization": True,
            "access_controls": True,
            "vulnerability_scanning": True,
            "monitoring": True,
            "performance_tracking": True
        }
    
    @pytest.mark.asyncio
    async def test_create_governance_framework(self, governance_engine, sample_governance_config):
        """Test creating AI governance framework"""
        framework = await governance_engine.create_governance_framework(
            name=sample_governance_config["name"],
            description=sample_governance_config["description"],
            policies=sample_governance_config["policies"],
            risk_thresholds=sample_governance_config["risk_thresholds"]
        )
        
        assert framework is not None
        assert framework.name == sample_governance_config["name"]
        assert framework.governance_policies == sample_governance_config["policies"]
        assert framework.risk_thresholds == sample_governance_config["risk_thresholds"]
    
    @pytest.mark.asyncio
    async def test_ai_safety_assessment(self, governance_engine, sample_ai_system_config):
        """Test comprehensive AI safety assessment"""
        deployment_context = {
            "environment": "production",
            "user_base": "global",
            "data_sensitivity": "high",
            "secure_deployment": True,
            "redundancy": True
        }
        
        assessment = await governance_engine.assess_ai_safety(
            ai_system_id=sample_ai_system_config["system_id"],
            system_config=sample_ai_system_config,
            deployment_context=deployment_context
        )
        
        assert assessment is not None
        assert "safety_scores" in assessment
        assert "risk_factors" in assessment
        assert "mitigation_strategies" in assessment
        assert "overall_safety_score" in assessment
        assert "risk_level" in assessment
        
        # Verify safety dimensions are assessed
        safety_scores = assessment["safety_scores"]
        expected_dimensions = ["robustness", "interpretability", "fairness", "privacy", "security", "reliability"]
        for dimension in expected_dimensions:
            assert dimension in safety_scores
            assert 0.0 <= safety_scores[dimension] <= 1.0
        
        # Verify overall safety score is reasonable
        assert 0.0 <= assessment["overall_safety_score"] <= 1.0
        assert assessment["risk_level"] in [level.value for level in RiskLevel]
    
    @pytest.mark.asyncio
    async def test_ai_alignment_evaluation(self, governance_engine):
        """Test AI system alignment evaluation"""
        objectives = [
            "Maximize user satisfaction",
            "Ensure fair treatment across demographics",
            "Protect user privacy",
            "Maintain system reliability"
        ]
        
        behavior_data = {
            "user_satisfaction_metrics": {"average_rating": 4.2, "completion_rate": 0.87},
            "fairness_metrics": {"demographic_parity": 0.92, "equal_opportunity": 0.89},
            "privacy_metrics": {"data_minimization_score": 0.85, "consent_rate": 0.94},
            "reliability_metrics": {"uptime": 0.999, "error_rate": 0.001}
        }
        
        evaluation = await governance_engine.evaluate_alignment(
            ai_system_id="test_system_001",
            objectives=objectives,
            behavior_data=behavior_data
        )
        
        assert evaluation is not None
        assert "objective_alignment" in evaluation
        assert "behavior_analysis" in evaluation
        assert "alignment_score" in evaluation
        assert "recommendations" in evaluation
        
        # Verify alignment with each objective is assessed
        objective_alignment = evaluation["objective_alignment"]
        for objective in objectives:
            assert objective in objective_alignment
        
        assert 0.0 <= evaluation["alignment_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_governance_compliance_monitoring(self, governance_engine, sample_governance_config):
        """Test ongoing governance compliance monitoring"""
        # First create a governance framework
        framework = await governance_engine.create_governance_framework(
            name=sample_governance_config["name"],
            description=sample_governance_config["description"],
            policies=sample_governance_config["policies"],
            risk_thresholds=sample_governance_config["risk_thresholds"]
        )
        
        # Monitor compliance
        monitoring_period = timedelta(hours=24)
        compliance_report = await governance_engine.monitor_governance_compliance(
            governance_id=str(framework.id),
            monitoring_period=monitoring_period
        )
        
        assert compliance_report is not None
        assert "compliance_metrics" in compliance_report
        assert "violations" in compliance_report
        assert "recommendations" in compliance_report
        assert "overall_compliance" in compliance_report
        
        # Verify compliance metrics structure
        compliance_metrics = compliance_report["compliance_metrics"]
        expected_metrics = ["policy", "safety", "ethical"]
        for metric in expected_metrics:
            assert metric in compliance_metrics
        
        assert 0.0 <= compliance_report["overall_compliance"] <= 1.0


class TestRegulatoryCompliance:
    """Test regulatory compliance capabilities"""
    
    @pytest.fixture
    def compliance_engine(self):
        return RegulatoryComplianceEngine()
    
    @pytest.fixture
    def sample_system_config(self):
        return {
            "data_encryption": True,
            "access_controls": True,
            "data_minimization": True,
            "consent_management": True,
            "explainable_ai": True,
            "algorithm_documentation": True,
            "decision_logging": True,
            "bias_testing": True,
            "fairness_metrics": True,
            "bias_monitoring": True,
            "human_review": True,
            "human_override": True,
            "escalation_procedures": True
        }
    
    @pytest.fixture
    def sample_data_processing_activities(self):
        return [
            {
                "activity_type": "user_profiling",
                "involves_personal_data": True,
                "involves_ai_decision_making": True,
                "data_categories": ["behavioral", "demographic"],
                "processing_purposes": ["personalization", "recommendation"]
            },
            {
                "activity_type": "content_moderation",
                "involves_personal_data": False,
                "involves_ai_decision_making": True,
                "data_categories": ["content"],
                "processing_purposes": ["safety", "compliance"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_global_compliance_assessment(
        self, 
        compliance_engine, 
        sample_system_config, 
        sample_data_processing_activities
    ):
        """Test global regulatory compliance assessment"""
        deployment_regions = ["EU", "US", "CA", "SG"]
        
        assessment = await compliance_engine.assess_global_compliance(
            ai_system_id="test_system_001",
            system_config=sample_system_config,
            deployment_regions=deployment_regions,
            data_processing_activities=sample_data_processing_activities
        )
        
        assert assessment is not None
        assert "applicable_regulations" in assessment
        assert "compliance_status" in assessment
        assert "compliance_gaps" in assessment
        assert "remediation_plan" in assessment
        assert "overall_compliance_score" in assessment
        
        # Verify applicable regulations are identified
        applicable_regulations = assessment["applicable_regulations"]
        assert len(applicable_regulations) > 0
        
        # Verify compliance status for each regulation
        compliance_status = assessment["compliance_status"]
        for regulation in applicable_regulations:
            assert regulation in compliance_status
            reg_status = compliance_status[regulation]
            assert "overall_status" in reg_status
            assert "compliance_score" in reg_status
            assert "requirement_compliance" in reg_status
            assert reg_status["overall_status"] in [status.value for status in ComplianceStatus]
        
        assert 0.0 <= assessment["overall_compliance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_assessment(
        self, 
        compliance_engine, 
        sample_system_config, 
        sample_data_processing_activities
    ):
        """Test GDPR-specific compliance assessment"""
        deployment_regions = ["EU"]
        
        assessment = await compliance_engine.assess_global_compliance(
            ai_system_id="gdpr_test_system",
            system_config=sample_system_config,
            deployment_regions=deployment_regions,
            data_processing_activities=sample_data_processing_activities
        )
        
        # Verify GDPR is identified as applicable
        assert Regulation.GDPR.value in assessment["applicable_regulations"]
        
        # Verify GDPR compliance assessment
        gdpr_compliance = assessment["compliance_status"][Regulation.GDPR.value]
        assert gdpr_compliance is not None
        assert "requirement_compliance" in gdpr_compliance
        
        # Check specific GDPR requirements
        requirements = gdpr_compliance["requirement_compliance"]
        expected_requirements = ["data_protection", "consent_management", "user_rights"]
        for req in expected_requirements:
            if req in requirements:
                assert "score" in requirements[req]
                assert "evidence" in requirements[req]
                assert "gaps" in requirements[req]
    
    @pytest.mark.asyncio
    async def test_eu_ai_act_compliance(
        self, 
        compliance_engine, 
        sample_system_config, 
        sample_data_processing_activities
    ):
        """Test EU AI Act compliance assessment"""
        deployment_regions = ["EU"]
        
        # Configure system as high-risk AI system
        high_risk_config = {
            **sample_system_config,
            "high_risk_application": True,
            "risk_assessment": True,
            "impact_assessment": True
        }
        
        assessment = await compliance_engine.assess_global_compliance(
            ai_system_id="eu_ai_act_test_system",
            system_config=high_risk_config,
            deployment_regions=deployment_regions,
            data_processing_activities=sample_data_processing_activities
        )
        
        # Verify EU AI Act is identified as applicable
        assert Regulation.EU_AI_ACT.value in assessment["applicable_regulations"]
        
        # Verify EU AI Act compliance assessment
        ai_act_compliance = assessment["compliance_status"][Regulation.EU_AI_ACT.value]
        assert ai_act_compliance is not None
        
        # Check specific AI Act requirements
        requirements = ai_act_compliance["requirement_compliance"]
        expected_requirements = ["risk_assessment", "algorithmic_transparency", "bias_assessment", "human_oversight"]
        for req in expected_requirements:
            if req in requirements:
                assert "score" in requirements[req]
                assert requirements[req]["score"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_regulatory_change_monitoring(self, compliance_engine):
        """Test regulatory change monitoring"""
        regions = ["EU", "US", "CA"]
        monitoring_period = timedelta(days=7)
        
        updates = await compliance_engine.monitor_regulatory_changes(
            regions=regions,
            monitoring_period=monitoring_period
        )
        
        assert updates is not None
        assert "regulatory_changes" in updates
        assert "impact_assessment" in updates
        assert "recommended_actions" in updates
        assert updates["regions"] == regions
    
    @pytest.mark.asyncio
    async def test_automated_compliance_reporting(self, compliance_engine):
        """Test automated compliance reporting"""
        system_data = {
            "data_processing_records": 1000000,
            "user_consent_rate": 0.95,
            "data_breach_incidents": 0,
            "user_rights_requests": 150,
            "response_time_average": 2.5,  # days
            "algorithm_decisions": 5000000,
            "human_review_rate": 0.15,
            "bias_audit_results": {"demographic_parity": 0.92, "equal_opportunity": 0.89}
        }
        
        report = await compliance_engine.automate_compliance_reporting(
            regulation=Regulation.GDPR.value,
            reporting_period="Q1_2024",
            system_data=system_data
        )
        
        assert report is not None
        assert "compliance_metrics" in report
        assert "violations" in report
        assert "corrective_actions" in report
        assert "certification_status" in report
        assert report["regulation"] == Regulation.GDPR.value


class TestEthicalDecisionMaking:
    """Test ethical decision-making capabilities"""
    
    @pytest.fixture
    def ethics_engine(self):
        return EthicalDecisionEngine()
    
    @pytest.fixture
    def sample_ethics_framework_config(self):
        return {
            "name": "Corporate AI Ethics Framework",
            "description": "Comprehensive ethics framework for AI development",
            "ethical_principles": [
                EthicalPrinciple.FAIRNESS.value,
                EthicalPrinciple.TRANSPARENCY.value,
                EthicalPrinciple.ACCOUNTABILITY.value,
                EthicalPrinciple.PRIVACY.value,
                EthicalPrinciple.SAFETY.value
            ],
            "decision_criteria": {
                "stakeholder_impact_weight": 0.4,
                "principle_adherence_weight": 0.6,
                "minimum_approval_score": 0.7
            },
            "stakeholder_considerations": {
                "end_users": 0.3,
                "employees": 0.2,
                "society": 0.25,
                "shareholders": 0.15,
                "regulators": 0.1
            }
        }
    
    @pytest.mark.asyncio
    async def test_create_ethics_framework(self, ethics_engine, sample_ethics_framework_config):
        """Test creating ethics framework"""
        framework = await ethics_engine.create_ethics_framework(
            name=sample_ethics_framework_config["name"],
            description=sample_ethics_framework_config["description"],
            ethical_principles=sample_ethics_framework_config["ethical_principles"],
            decision_criteria=sample_ethics_framework_config["decision_criteria"],
            stakeholder_considerations=sample_ethics_framework_config["stakeholder_considerations"]
        )
        
        assert framework is not None
        assert framework.name == sample_ethics_framework_config["name"]
        assert framework.ethical_principles == sample_ethics_framework_config["ethical_principles"]
    
    @pytest.mark.asyncio
    async def test_ethical_decision_evaluation(self, ethics_engine, sample_ethics_framework_config):
        """Test ethical decision evaluation"""
        # Create framework first
        framework = await ethics_engine.create_ethics_framework(**sample_ethics_framework_config)
        
        decision_context = {
            "decision_type": "algorithm_deployment",
            "scope": "global",
            "urgency": "medium",
            "reversibility": "high"
        }
        
        proposed_action = {
            "action": "deploy_recommendation_algorithm",
            "bias_mitigation": True,
            "explainable": True,
            "documented": True,
            "data_minimization": True,
            "risk_assessment": True,
            "human_control": True
        }
        
        stakeholder_impacts = {
            "end_users": {"impact_type": "positive", "magnitude": 0.8},
            "employees": {"impact_type": "neutral", "magnitude": 0.1},
            "society": {"impact_type": "positive", "magnitude": 0.6},
            "shareholders": {"impact_type": "positive", "magnitude": 0.7}
        }
        
        evaluation = await ethics_engine.evaluate_ethical_decision(
            framework_id=str(framework.id),
            decision_context=decision_context,
            proposed_action=proposed_action,
            stakeholder_impacts=stakeholder_impacts
        )
        
        assert evaluation is not None
        assert "principle_scores" in evaluation
        assert "stakeholder_analysis" in evaluation
        assert "ethical_risks" in evaluation
        assert "recommendations" in evaluation
        assert "overall_score" in evaluation
        assert "decision_outcome" in evaluation
        
        # Verify principle scores
        principle_scores = evaluation["principle_scores"]
        for principle in sample_ethics_framework_config["ethical_principles"]:
            assert principle in principle_scores
            assert 0.0 <= principle_scores[principle] <= 1.0
        
        assert 0.0 <= evaluation["overall_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_ethical_dilemma_resolution(self, ethics_engine):
        """Test ethical dilemma resolution"""
        dilemma_type = "privacy_vs_utility"
        conflicting_principles = [EthicalPrinciple.PRIVACY.value, EthicalPrinciple.SAFETY.value]
        
        context = {
            "scenario": "contact_tracing_app",
            "privacy_concerns": "location_tracking",
            "utility_benefits": "disease_prevention",
            "affected_population": "general_public"
        }
        
        stakeholder_preferences = {
            "public_health_officials": {"priority": EthicalPrinciple.SAFETY.value, "weight": 0.8},
            "privacy_advocates": {"priority": EthicalPrinciple.PRIVACY.value, "weight": 0.9},
            "general_public": {"priority": "balanced", "weight": 0.6}
        }
        
        resolution = await ethics_engine.resolve_ethical_dilemma(
            dilemma_type=dilemma_type,
            conflicting_principles=conflicting_principles,
            context=context,
            stakeholder_preferences=stakeholder_preferences
        )
        
        assert resolution is not None
        assert "analysis" in resolution
        assert "trade_offs" in resolution
        assert "recommended_approach" in resolution
        assert "alternative_solutions" in resolution
        assert resolution["dilemma_type"] == dilemma_type
        assert resolution["conflicting_principles"] == conflicting_principles
    
    @pytest.mark.asyncio
    async def test_stakeholder_analysis(self, ethics_engine):
        """Test comprehensive stakeholder analysis"""
        decision_context = {
            "decision": "implement_ai_hiring_system",
            "scope": "company_wide",
            "timeline": "6_months"
        }
        
        proposed_changes = [
            {
                "change": "automated_resume_screening",
                "impact_areas": ["efficiency", "bias_reduction", "job_displacement"]
            },
            {
                "change": "ai_interview_analysis",
                "impact_areas": ["consistency", "privacy", "human_interaction"]
            }
        ]
        
        analysis = await ethics_engine.conduct_stakeholder_analysis(
            decision_context=decision_context,
            proposed_changes=proposed_changes
        )
        
        assert analysis is not None
        assert "stakeholder_impacts" in analysis
        assert "impact_severity" in analysis
        assert "mitigation_strategies" in analysis
        assert "engagement_recommendations" in analysis
        
        # Verify stakeholder impacts are analyzed
        stakeholder_impacts = analysis["stakeholder_impacts"]
        expected_stakeholders = ["end_users", "employees", "customers", "society"]
        for stakeholder in expected_stakeholders:
            if stakeholder in stakeholder_impacts:
                assert isinstance(stakeholder_impacts[stakeholder], dict)


class TestPublicPolicyAnalysis:
    """Test public policy analysis capabilities"""
    
    @pytest.fixture
    def policy_engine(self):
        return PublicPolicyEngine()
    
    @pytest.mark.asyncio
    async def test_policy_landscape_analysis(self, policy_engine):
        """Test policy landscape analysis"""
        analysis_scope = {
            "focus_areas": ["ai_regulation", "data_protection"],
            "time_horizon": "2_years",
            "stakeholder_groups": ["government", "industry", "civil_society"]
        }
        
        analysis = await policy_engine.analyze_policy_landscape(
            policy_area=PolicyArea.AI_REGULATION.value,
            jurisdiction="EU",
            analysis_scope=analysis_scope
        )
        
        assert analysis is not None
        assert "current_policies" in analysis
        assert "policy_gaps" in analysis
        assert "stakeholder_positions" in analysis
        assert "regulatory_trends" in analysis
        assert "international_comparisons" in analysis
        assert "recommendations" in analysis
        assert analysis["policy_area"] == PolicyArea.AI_REGULATION.value
        assert analysis["jurisdiction"] == "EU"
    
    @pytest.mark.asyncio
    async def test_policy_strategy_development(self, policy_engine):
        """Test policy strategy development"""
        target_outcomes = [
            "Ensure AI safety and reliability",
            "Promote innovation and competitiveness",
            "Protect fundamental rights",
            "Enable regulatory clarity"
        ]
        
        constraints = {
            "budget_limitations": True,
            "political_feasibility": "medium",
            "international_coordination": "required",
            "implementation_timeline": "2_years"
        }
        
        stakeholder_requirements = {
            "industry": {"priority": "regulatory_clarity", "concern": "compliance_costs"},
            "civil_society": {"priority": "rights_protection", "concern": "enforcement"},
            "government": {"priority": "balanced_approach", "concern": "competitiveness"}
        }
        
        strategy = await policy_engine.develop_policy_strategy(
            policy_objective="Comprehensive AI Regulation Framework",
            target_outcomes=target_outcomes,
            constraints=constraints,
            stakeholder_requirements=stakeholder_requirements
        )
        
        assert strategy is not None
        assert "strategic_approach" in strategy
        assert "implementation_roadmap" in strategy
        assert "stakeholder_engagement_plan" in strategy
        assert "risk_mitigation" in strategy
        assert "success_metrics" in strategy
        assert "monitoring_framework" in strategy
        assert strategy["target_outcomes"] == target_outcomes
    
    @pytest.mark.asyncio
    async def test_policy_impact_assessment(self, policy_engine):
        """Test policy impact assessment"""
        proposed_policy = {
            "name": "AI Transparency and Accountability Act",
            "scope": "High-risk AI systems",
            "requirements": ["algorithmic_auditing", "transparency_reports", "human_oversight"],
            "enforcement_mechanism": "regulatory_fines"
        }
        
        affected_sectors = ["technology", "healthcare", "finance", "transportation"]
        
        implementation_timeline = {
            "preparation_phase": "6_months",
            "pilot_implementation": "12_months",
            "full_implementation": "24_months"
        }
        
        assessment = await policy_engine.assess_policy_impact(
            proposed_policy=proposed_policy,
            affected_sectors=affected_sectors,
            implementation_timeline=implementation_timeline
        )
        
        assert assessment is not None
        assert "economic_impact" in assessment
        assert "social_impact" in assessment
        assert "technological_impact" in assessment
        assert "regulatory_impact" in assessment
        assert "stakeholder_impact" in assessment
        assert "unintended_consequences" in assessment
        assert "mitigation_strategies" in assessment
        assert assessment["affected_sectors"] == affected_sectors
    
    @pytest.mark.asyncio
    async def test_policy_implementation_monitoring(self, policy_engine):
        """Test policy implementation monitoring"""
        implementation_metrics = {
            "compliance_rate": 0.75,
            "enforcement_actions": 12,
            "stakeholder_satisfaction": 0.68,
            "implementation_progress": 0.80,
            "budget_utilization": 0.85
        }
        
        monitoring_period = timedelta(days=30)
        
        report = await policy_engine.monitor_policy_implementation(
            policy_id="ai_transparency_act_001",
            implementation_metrics=implementation_metrics,
            monitoring_period=monitoring_period
        )
        
        assert report is not None
        assert "implementation_progress" in report
        assert "performance_metrics" in report
        assert "compliance_status" in report
        assert "stakeholder_feedback" in report
        assert "challenges_identified" in report
        assert "recommendations" in report
        assert report["policy_id"] == "ai_transparency_act_001"


if __name__ == "__main__":
    pytest.main([__file__])