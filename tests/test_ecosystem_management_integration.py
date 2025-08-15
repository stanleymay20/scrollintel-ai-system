"""
Integration Tests for Ecosystem Management

This module contains comprehensive integration tests for the ecosystem management
capabilities, including team optimization, partnership analysis, organizational
design, and global coordination.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.team_optimization_engine import TeamOptimizationEngine
from scrollintel.engines.partnership_analysis_engine import PartnershipAnalysisEngine
from scrollintel.engines.organizational_design_engine import OrganizationalDesignEngine
from scrollintel.engines.global_coordination_engine import GlobalCoordinationEngine
from scrollintel.models.ecosystem_models import (
    EngineerProfile, TeamMetrics, TeamOptimization, PartnershipOpportunity,
    PartnershipManagement, AcquisitionTarget, OrganizationalDesign,
    GlobalTeamCoordination, CommunicationOptimization, EcosystemHealthMetrics,
    TeamRole, ProductivityMetric, PartnershipType, AcquisitionStage
)


class TestEcosystemManagementIntegration:
    """Integration tests for ecosystem management capabilities"""
    
    @pytest.fixture
    def sample_engineers(self) -> List[EngineerProfile]:
        """Create sample engineer profiles for testing"""
        engineers = []
        
        # Create diverse engineer profiles across different teams and locations
        for i in range(100):  # 100 engineers for testing
            engineer = EngineerProfile(
                id=f"eng_{i:03d}",
                name=f"Engineer {i}",
                role=TeamRole.SENIOR_ENGINEER if i % 3 == 0 else TeamRole.STAFF_ENGINEER,
                team_id=f"team_{i // 10}",  # 10 teams of 10 engineers each
                location=["San Francisco", "New York", "London", "Singapore", "Sydney"][i % 5],
                timezone=[f"UTC{tz}" for tz in [-8, -5, 0, 8, 10]][i % 5],
                skills=["python", "javascript", "kubernetes", "machine_learning", "security"][i % 5:i % 5 + 3],
                experience_years=2 + (i % 15),  # 2-16 years experience
                productivity_metrics={
                    ProductivityMetric.FEATURES_DELIVERED: 0.6 + (i % 40) / 100,
                    ProductivityMetric.CODE_REVIEWS: 0.7 + (i % 30) / 100,
                    ProductivityMetric.INNOVATION_CONTRIBUTIONS: 0.5 + (i % 50) / 100
                },
                collaboration_score=0.6 + (i % 40) / 100,
                innovation_score=0.5 + (i % 50) / 100,
                mentoring_capacity=i % 5,
                current_projects=[f"project_{j}" for j in range(i % 3 + 1)],
                performance_trend=0.1 if i % 4 == 0 else -0.05 if i % 7 == 0 else 0.0,
                satisfaction_score=0.7 + (i % 30) / 100,
                retention_risk=0.1 + (i % 20) / 100
            )
            engineers.append(engineer)
        
        return engineers
    
    @pytest.fixture
    def sample_teams(self) -> List[TeamMetrics]:
        """Create sample team metrics for testing"""
        teams = []
        
        for i in range(10):  # 10 teams
            team = TeamMetrics(
                team_id=f"team_{i}",
                team_name=f"Team {i}",
                size=10,
                manager_id=f"manager_{i}",
                department=["Platform", "Product", "Data", "Research"][i % 4],
                location=["San Francisco", "New York", "London", "Singapore", "Sydney"][i % 5],
                productivity_score=0.6 + (i % 40) / 100,
                velocity=0.7 + (i % 30) / 100,
                quality_score=0.8 + (i % 20) / 100,
                collaboration_index=0.65 + (i % 35) / 100,
                innovation_rate=0.5 + (i % 50) / 100,
                technical_debt_ratio=0.2 + (i % 30) / 100,
                delivery_predictability=0.75 + (i % 25) / 100,
                team_satisfaction=0.7 + (i % 30) / 100,
                turnover_rate=0.05 + (i % 15) / 100,
                hiring_velocity=0.8 + (i % 20) / 100
            )
            teams.append(team)
        
        return teams
    
    @pytest.fixture
    def sample_partnerships(self) -> List[PartnershipManagement]:
        """Create sample partnerships for testing"""
        partnerships = []
        
        for i in range(5):  # 5 active partnerships
            partnership = PartnershipManagement(
                id=f"partnership_{i}",
                partner_id=f"partner_{i}",
                partnership_type=list(PartnershipType)[i % len(PartnershipType)],
                start_date=datetime.now() - timedelta(days=365 * (i + 1)),
                status="active",
                key_objectives=[f"objective_{j}" for j in range(3)],
                success_metrics={f"metric_{j}": 0.8 for j in range(3)},
                current_performance={f"metric_{j}": 0.7 + (i % 20) / 100 for j in range(3)},
                relationship_health=0.7 + (i % 30) / 100,
                communication_frequency=4 + (i % 3),
                joint_initiatives=[f"initiative_{j}" for j in range(2)],
                value_delivered=1000000 * (i + 1),
                challenges=[f"challenge_{j}" for j in range(i % 3 + 1)],
                next_milestones=[{"milestone": f"milestone_{j}", "date": datetime.now() + timedelta(days=30 * j)} for j in range(2)]
            )
            partnerships.append(partnership)
        
        return partnerships
    
    @pytest.mark.asyncio
    async def test_team_optimization_integration(self, sample_engineers, sample_teams):
        """Test complete team optimization workflow"""
        # Initialize engine
        optimizer = TeamOptimizationEngine()
        
        # Define optimization goals
        optimization_goals = {
            'productivity_increase': 0.20,
            'quality_improvement': 0.15,
            'innovation_boost': 0.25,
            'satisfaction_improvement': 0.10
        }
        
        # Run optimization
        optimizations = await optimizer.optimize_global_productivity(
            sample_engineers, sample_teams, optimization_goals
        )
        
        # Verify results
        assert len(optimizations) == len(sample_teams)
        assert all(isinstance(opt, TeamOptimization) for opt in optimizations)
        
        # Check that optimizations have required components
        for optimization in optimizations:
            assert optimization.team_id in [team.team_id for team in sample_teams]
            assert len(optimization.recommended_actions) > 0
            assert optimization.success_probability > 0
            assert optimization.roi_projection >= 0
            assert len(optimization.risk_factors) >= 0
            
            # Verify expected improvements are reasonable
            for metric, improvement in optimization.expected_improvements.items():
                assert 0 <= improvement <= 1.0  # Improvements should be between 0-100%
        
        # Verify high-performing teams get different recommendations than low-performing ones
        high_perf_teams = [team for team in sample_teams if team.productivity_score > 0.8]
        low_perf_teams = [team for team in sample_teams if team.productivity_score < 0.6]
        
        if high_perf_teams and low_perf_teams:
            high_perf_opts = [opt for opt in optimizations if opt.team_id in [t.team_id for t in high_perf_teams]]
            low_perf_opts = [opt for opt in optimizations if opt.team_id in [t.team_id for t in low_perf_teams]]
            
            # Low-performing teams should have more aggressive optimization recommendations
            avg_high_perf_actions = sum(len(opt.recommended_actions) for opt in high_perf_opts) / len(high_perf_opts)
            avg_low_perf_actions = sum(len(opt.recommended_actions) for opt in low_perf_opts) / len(low_perf_opts)
            
            assert avg_low_perf_actions >= avg_high_perf_actions
    
    @pytest.mark.asyncio
    async def test_partnership_analysis_integration(self):
        """Test complete partnership analysis workflow"""
        # Initialize engine
        analyzer = PartnershipAnalysisEngine()
        
        # Define strategic goals and context
        strategic_goals = {
            'technology_expansion': ['ai_ml', 'quantum_computing', 'edge_computing'],
            'market_expansion': ['asia_pacific', 'europe'],
            'research_advancement': ['fundamental_ai', 'quantum_research']
        }
        
        market_context = {
            'growth_sectors': ['ai', 'cloud', 'quantum'],
            'competitive_pressure': 0.8,
            'innovation_pace': 0.9
        }
        
        current_capabilities = ['cloud_infrastructure', 'machine_learning', 'data_analytics']
        
        # Analyze partnership opportunities
        opportunities = await analyzer.analyze_partnership_opportunities(
            strategic_goals, market_context, current_capabilities
        )
        
        # Verify results
        assert len(opportunities) > 0
        assert all(isinstance(opp, PartnershipOpportunity) for opp in opportunities)
        
        # Check that opportunities are properly ranked
        strategic_values = [opp.strategic_value for opp in opportunities]
        assert strategic_values == sorted(strategic_values, reverse=True)
        
        # Verify opportunity components
        for opportunity in opportunities:
            assert 0 <= opportunity.strategic_value <= 1.0
            assert 0 <= opportunity.technology_synergy <= 1.0
            assert 0 <= opportunity.revenue_potential <= 1.0
            assert opportunity.timeline_to_value > 0
            assert len(opportunity.risk_assessment) > 0
    
    @pytest.mark.asyncio
    async def test_acquisition_analysis_integration(self):
        """Test complete acquisition analysis workflow"""
        # Initialize engine
        analyzer = PartnershipAnalysisEngine()
        
        # Define acquisition strategy
        acquisition_strategy = {
            'technology_focus': ['ai_research', 'quantum_computing', 'edge_ai'],
            'talent_needs': ['ai_researchers', 'quantum_engineers'],
            'target_markets': ['enterprise_ai', 'quantum_cloud']
        }
        
        budget_constraints = {
            'max_valuation': 5000000000,  # $5B max
            'available_budget': 10000000000  # $10B available
        }
        
        strategic_priorities = ['technology_acquisition', 'talent_acquisition']
        
        # Analyze acquisition targets
        targets = await analyzer.analyze_acquisition_targets(
            acquisition_strategy, budget_constraints, strategic_priorities
        )
        
        # Verify results
        assert len(targets) > 0
        assert all(isinstance(target, AcquisitionTarget) for target in targets)
        
        # Check that all targets are within budget
        for target in targets:
            assert target.valuation <= budget_constraints['max_valuation']
            assert target.strategic_fit > 0.6  # Should meet minimum strategic fit
            assert 0 <= target.integration_risk <= 1.0
            assert 0 <= target.synergy_potential <= 1.0
    
    @pytest.mark.asyncio
    async def test_partnership_management_integration(self, sample_partnerships):
        """Test active partnership management workflow"""
        # Initialize engine
        analyzer = PartnershipAnalysisEngine()
        
        # Manage active partnerships
        management_results = await analyzer.manage_active_partnerships(sample_partnerships)
        
        # Verify results structure
        assert 'partnership_health' in management_results
        assert 'recommendations' in management_results
        assert 'portfolio_metrics' in management_results
        assert 'risk_alerts' in management_results
        
        # Check partnership health assessments
        health_assessments = management_results['partnership_health']
        assert len(health_assessments) == len(sample_partnerships)
        
        for partnership_id, health in health_assessments.items():
            assert 'overall_score' in health
            assert 0 <= health['overall_score'] <= 1.0
            assert 'trend' in health
        
        # Check portfolio metrics
        portfolio_metrics = management_results['portfolio_metrics']
        assert 'portfolio_health' in portfolio_metrics
        assert 'total_partnerships' in portfolio_metrics
        assert portfolio_metrics['total_partnerships'] == len(sample_partnerships)
        
        # Verify recommendations are generated for unhealthy partnerships
        recommendations = management_results['recommendations']
        unhealthy_partnerships = [
            p for p in sample_partnerships 
            if health_assessments[p.id]['overall_score'] < 0.7
        ]
        
        if unhealthy_partnerships:
            assert len(recommendations) > 0
            for rec in recommendations:
                assert 'type' in rec
                assert 'priority' in rec
                assert 'expected_impact' in rec
    
    @pytest.mark.asyncio
    async def test_organizational_design_integration(self, sample_engineers, sample_teams):
        """Test organizational design optimization workflow"""
        # Initialize engine
        designer = OrganizationalDesignEngine()
        
        # Define current structure and objectives
        current_structure = {
            'levels': 5,
            'departments': ['Platform', 'Product', 'Data', 'Research'],
            'total_managers': 25,
            'span_of_control': {'average': 8, 'max': 15, 'min': 3}
        }
        
        business_objectives = {
            'priority': 'speed',
            'growth_target': 2.0,  # 2x growth
            'innovation_focus': True,
            'global_expansion': True
        }
        
        # Optimize organizational structure
        org_design = await designer.optimize_organizational_structure(
            current_structure, sample_engineers, sample_teams, business_objectives
        )
        
        # Verify results
        assert isinstance(org_design, OrganizationalDesign)
        assert org_design.current_structure == current_structure
        assert 'hierarchy' in org_design.recommended_structure
        assert 'functional_groups' in org_design.recommended_structure
        
        # Check implementation plan
        assert len(org_design.implementation_plan) > 0
        for phase in org_design.implementation_plan:
            assert 'phase' in phase
            assert 'activities' in phase
            assert 'success_criteria' in phase
            assert 'risks' in phase
        
        # Verify expected benefits
        assert len(org_design.expected_benefits) > 0
        for benefit, value in org_design.expected_benefits.items():
            assert isinstance(value, (int, float))
        
        # Check risk mitigation strategies
        assert len(org_design.risk_mitigation) > 0
    
    @pytest.mark.asyncio
    async def test_global_coordination_integration(self, sample_engineers, sample_teams):
        """Test global coordination optimization workflow"""
        # Initialize engine
        coordinator = GlobalCoordinationEngine()
        
        # Define coordination constraints
        coordination_constraints = {
            'regulatory_requirements': ['gdpr', 'ccpa'],
            'business_hours_overlap': 4,  # minimum 4 hours overlap
            'critical_timezones': ['UTC-8', 'UTC+0', 'UTC+8'],
            'language_requirements': ['english', 'mandarin']
        }
        
        # Optimize global coordination
        coordination = await coordinator.optimize_global_coordination(
            sample_engineers, sample_teams, coordination_constraints
        )
        
        # Verify results
        assert isinstance(coordination, GlobalTeamCoordination)
        assert coordination.total_engineers == len(sample_engineers)
        assert coordination.active_teams == len(sample_teams)
        assert len(coordination.global_locations) > 0
        
        # Check coordination metrics
        assert 0 <= coordination.communication_efficiency <= 1.0
        assert 0 <= coordination.coordination_overhead <= 1.0
        assert 0 <= coordination.global_velocity <= 1.0
        assert 0 <= coordination.knowledge_sharing_index <= 1.0
        assert 0 <= coordination.cultural_alignment_score <= 1.0
        
        # Verify timezone coverage
        assert len(coordination.timezone_coverage) > 0
        
        # Check cross-team dependencies
        assert len(coordination.cross_team_dependencies) > 0
    
    @pytest.mark.asyncio
    async def test_communication_optimization_integration(self, sample_engineers, sample_teams):
        """Test communication optimization workflow"""
        # Initialize engine
        coordinator = GlobalCoordinationEngine()
        
        # Define current communication setup
        current_communication = {
            'tools': ['slack', 'zoom', 'email', 'jira', 'confluence', 'github'],
            'usage_patterns': {
                'slack': {'daily_users': 8000, 'messages_per_day': 50000},
                'zoom': {'daily_meetings': 500, 'avg_duration': 45},
                'email': {'daily_emails': 20000}
            },
            'satisfaction_scores': {
                'slack': 0.8,
                'zoom': 0.7,
                'email': 0.5
            }
        }
        
        # Optimize communication
        comm_optimization = await coordinator.optimize_communication_systems(
            sample_engineers, sample_teams, current_communication
        )
        
        # Verify results
        assert isinstance(comm_optimization, CommunicationOptimization)
        assert comm_optimization.current_communication_patterns == current_communication
        
        # Check inefficiencies identification
        assert len(comm_optimization.inefficiencies_identified) >= 0
        
        # Verify optimization recommendations
        assert len(comm_optimization.optimization_recommendations) >= 0
        for rec in comm_optimization.optimization_recommendations:
            assert 'type' in rec
            assert 'priority' in rec
            assert 'expected_impact' in rec
        
        # Check tool recommendations
        assert len(comm_optimization.tool_recommendations) > 0
        for tool in comm_optimization.tool_recommendations:
            assert 'category' in tool
            assert 'tool' in tool
            assert 'rationale' in tool
        
        # Verify ROI projection
        assert comm_optimization.roi_projection >= 0
        assert comm_optimization.implementation_cost > 0
    
    @pytest.mark.asyncio
    async def test_ecosystem_health_monitoring_integration(self, sample_engineers, sample_teams, sample_partnerships):
        """Test comprehensive ecosystem health monitoring"""
        # Initialize engine
        coordinator = GlobalCoordinationEngine()
        
        # Create mock global coordination
        global_coordination = GlobalTeamCoordination(
            id="test_coord",
            timestamp=datetime.now(),
            total_engineers=len(sample_engineers),
            active_teams=len(sample_teams),
            global_locations=list(set(e.location for e in sample_engineers)),
            timezone_coverage={tz: 20 for tz in set(e.timezone for e in sample_engineers)},
            cross_team_dependencies={f"team_{i}": [f"team_{j}"] for i in range(5) for j in range(2)},
            communication_efficiency=0.75,
            coordination_overhead=0.25,
            global_velocity=0.80,
            knowledge_sharing_index=0.70,
            cultural_alignment_score=0.85,
            language_barriers={'english': 0.8, 'other': 0.2}
        )
        
        # Monitor ecosystem health
        health_metrics = await coordinator.monitor_ecosystem_health(
            sample_engineers, sample_teams, sample_partnerships, global_coordination
        )
        
        # Verify results
        assert isinstance(health_metrics, EcosystemHealthMetrics)
        assert health_metrics.total_engineers == len(sample_engineers)
        
        # Check all health metrics are within valid ranges
        assert 0 <= health_metrics.productivity_index <= 1.0
        assert 0 <= health_metrics.innovation_rate <= 1.0
        assert 0 <= health_metrics.collaboration_score <= 1.0
        assert 0 <= health_metrics.retention_rate <= 1.0
        assert 0 <= health_metrics.hiring_success_rate <= 1.0
        assert 0 <= health_metrics.organizational_agility <= 1.0
        assert 0 <= health_metrics.overall_health_score <= 1.0
        
        # Verify trend indicators
        assert len(health_metrics.trend_indicators) > 0
        for indicator, value in health_metrics.trend_indicators.items():
            assert isinstance(value, (int, float))
        
        # Check risk factors and improvement opportunities
        assert isinstance(health_metrics.risk_factors, list)
        assert isinstance(health_metrics.improvement_opportunities, list)
    
    @pytest.mark.asyncio
    async def test_end_to_end_ecosystem_optimization(self, sample_engineers, sample_teams, sample_partnerships):
        """Test complete end-to-end ecosystem optimization workflow"""
        # Initialize all engines
        team_optimizer = TeamOptimizationEngine()
        partnership_analyzer = PartnershipAnalysisEngine()
        org_designer = OrganizationalDesignEngine()
        global_coordinator = GlobalCoordinationEngine()
        
        # Step 1: Optimize teams
        optimization_goals = {'productivity_increase': 0.20, 'quality_improvement': 0.15}
        team_optimizations = await team_optimizer.optimize_global_productivity(
            sample_engineers, sample_teams, optimization_goals
        )
        
        # Step 2: Analyze partnerships
        strategic_goals = {'technology_expansion': ['ai_ml'], 'market_expansion': ['asia']}
        market_context = {'growth_sectors': ['ai'], 'competitive_pressure': 0.8}
        current_capabilities = ['cloud_infrastructure', 'machine_learning']
        
        partnership_opportunities = await partnership_analyzer.analyze_partnership_opportunities(
            strategic_goals, market_context, current_capabilities
        )
        
        # Step 3: Optimize organizational structure
        current_structure = {'levels': 5, 'departments': ['Platform', 'Product']}
        business_objectives = {'priority': 'speed', 'growth_target': 2.0}
        
        org_design = await org_designer.optimize_organizational_structure(
            current_structure, sample_engineers, sample_teams, business_objectives
        )
        
        # Step 4: Optimize global coordination
        coordination_constraints = {'business_hours_overlap': 4}
        global_coordination = await global_coordinator.optimize_global_coordination(
            sample_engineers, sample_teams, coordination_constraints
        )
        
        # Step 5: Monitor overall health
        health_metrics = await global_coordinator.monitor_ecosystem_health(
            sample_engineers, sample_teams, sample_partnerships, global_coordination
        )
        
        # Verify end-to-end integration
        assert len(team_optimizations) == len(sample_teams)
        assert len(partnership_opportunities) > 0
        assert isinstance(org_design, OrganizationalDesign)
        assert isinstance(global_coordination, GlobalTeamCoordination)
        assert isinstance(health_metrics, EcosystemHealthMetrics)
        
        # Verify consistency across optimizations
        assert health_metrics.total_engineers == len(sample_engineers)
        assert global_coordination.total_engineers == len(sample_engineers)
        assert global_coordination.active_teams == len(sample_teams)
        
        # Check that optimizations are coherent
        # Teams with higher optimization ROI should contribute to better overall health
        high_roi_optimizations = [opt for opt in team_optimizations if opt.roi_projection > 2.0]
        if high_roi_optimizations:
            # Overall health should be reasonable if we have high-ROI optimizations available
            assert health_metrics.overall_health_score > 0.5
    
    @pytest.mark.asyncio
    async def test_performance_at_scale(self):
        """Test ecosystem management performance with large datasets"""
        # Create large-scale test data
        large_engineer_count = 10000
        large_team_count = 1000
        
        # Generate large engineer dataset
        large_engineers = []
        for i in range(large_engineer_count):
            engineer = EngineerProfile(
                id=f"eng_{i:05d}",
                name=f"Engineer {i}",
                role=TeamRole.SENIOR_ENGINEER,
                team_id=f"team_{i // 10}",
                location=["SF", "NYC", "LON", "SG", "SYD"][i % 5],
                timezone=[f"UTC{tz}" for tz in [-8, -5, 0, 8, 10]][i % 5],
                skills=["python", "javascript", "kubernetes"][i % 3:i % 3 + 2],
                experience_years=5,
                productivity_metrics={ProductivityMetric.FEATURES_DELIVERED: 0.7},
                collaboration_score=0.7,
                innovation_score=0.6,
                mentoring_capacity=2,
                current_projects=["project_1"],
                performance_trend=0.0,
                satisfaction_score=0.8,
                retention_risk=0.1
            )
            large_engineers.append(engineer)
        
        # Generate large team dataset
        large_teams = []
        for i in range(large_team_count):
            team = TeamMetrics(
                team_id=f"team_{i}",
                team_name=f"Team {i}",
                size=10,
                manager_id=f"manager_{i}",
                department="Engineering",
                location="SF",
                productivity_score=0.7,
                velocity=0.8,
                quality_score=0.8,
                collaboration_index=0.7,
                innovation_rate=0.6,
                technical_debt_ratio=0.2,
                delivery_predictability=0.8,
                team_satisfaction=0.8,
                turnover_rate=0.05,
                hiring_velocity=0.9
            )
            large_teams.append(team)
        
        # Test team optimization at scale
        optimizer = TeamOptimizationEngine()
        start_time = datetime.now()
        
        optimizations = await optimizer.optimize_global_productivity(
            large_engineers, large_teams, {'productivity_increase': 0.15}
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Verify performance requirements
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert len(optimizations) == large_team_count
        
        # Test global coordination at scale
        coordinator = GlobalCoordinationEngine()
        start_time = datetime.now()
        
        coordination = await coordinator.optimize_global_coordination(
            large_engineers, large_teams, {'business_hours_overlap': 4}
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Verify performance requirements
        assert execution_time < 20.0  # Should complete within 20 seconds
        assert coordination.total_engineers == large_engineer_count
        assert coordination.active_teams == large_team_count
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, sample_engineers, sample_teams):
        """Test error handling and system resilience"""
        # Test with invalid data
        optimizer = TeamOptimizationEngine()
        
        # Test with empty data
        empty_optimizations = await optimizer.optimize_global_productivity(
            [], [], {'productivity_increase': 0.15}
        )
        assert len(empty_optimizations) == 0
        
        # Test with malformed goals
        try:
            await optimizer.optimize_global_productivity(
                sample_engineers, sample_teams, {}  # Empty goals
            )
            # Should handle gracefully, not crash
        except Exception as e:
            # If it raises an exception, it should be a handled one
            assert "optimization" in str(e).lower() or "goal" in str(e).lower()
        
        # Test partnership analyzer with invalid data
        analyzer = PartnershipAnalysisEngine()
        
        try:
            opportunities = await analyzer.analyze_partnership_opportunities(
                {}, {}, []  # All empty
            )
            # Should return empty list or handle gracefully
            assert isinstance(opportunities, list)
        except Exception as e:
            # Should be a handled exception
            assert "partnership" in str(e).lower() or "analysis" in str(e).lower()
        
        # Test organizational designer with inconsistent data
        designer = OrganizationalDesignEngine()
        
        # Create inconsistent data (engineers referencing non-existent teams)
        inconsistent_engineers = [
            EngineerProfile(
                id="eng_001",
                name="Engineer 1",
                role=TeamRole.SENIOR_ENGINEER,
                team_id="nonexistent_team",  # This team doesn't exist
                location="SF",
                timezone="UTC-8",
                skills=["python"],
                experience_years=5,
                productivity_metrics={ProductivityMetric.FEATURES_DELIVERED: 0.7},
                collaboration_score=0.7,
                innovation_score=0.6,
                mentoring_capacity=2,
                current_projects=["project_1"],
                performance_trend=0.0,
                satisfaction_score=0.8,
                retention_risk=0.1
            )
        ]
        
        try:
            org_design = await designer.optimize_organizational_structure(
                {'levels': 3}, inconsistent_engineers, sample_teams, {'priority': 'speed'}
            )
            # Should handle gracefully
            assert isinstance(org_design, OrganizationalDesign)
        except Exception as e:
            # Should be a handled exception
            assert "organization" in str(e).lower() or "structure" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])