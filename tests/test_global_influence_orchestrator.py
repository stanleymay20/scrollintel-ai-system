"""
Tests for Global Influence Network Orchestrator
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.global_influence_orchestrator import GlobalInfluenceOrchestrator
from scrollintel.models.global_influence_models import (
    InfluenceCampaign, InfluenceTarget, InfluenceNetwork,
    CampaignPriority, CampaignStatus, InfluenceScope, StakeholderType
)


class TestGlobalInfluenceOrchestrator:
    """Test suite for Global Influence Orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return GlobalInfluenceOrchestrator()
    
    @pytest.fixture
    def sample_campaign_data(self):
        """Sample campaign data for testing"""
        return {
            'objective': 'Establish AI technology leadership in healthcare',
            'target_outcomes': [
                'Gain recognition as healthcare AI thought leader',
                'Build partnerships with top medical institutions',
                'Influence healthcare AI policy discussions'
            ],
            'timeline': timedelta(days=120),
            'priority': 'high'
        }
    
    @pytest.fixture
    def sample_constraints(self):
        """Sample constraints for testing"""
        return {
            'budget_limit': 500000,
            'geographic_restrictions': ['US', 'EU'],
            'compliance_requirements': ['HIPAA', 'GDPR'],
            'timeline_flexibility': 'low'
        }
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'relationship_engine')
        assert hasattr(orchestrator, 'influence_engine')
        assert hasattr(orchestrator, 'partnership_engine')
        assert orchestrator.active_campaigns == {}
        assert orchestrator.network_state == {}
        assert orchestrator.influence_metrics == {}
    
    @pytest.mark.asyncio
    async def test_orchestrate_global_influence_campaign(self, orchestrator, sample_campaign_data):
        """Test campaign orchestration"""
        result = await orchestrator.orchestrate_global_influence_campaign(
            campaign_objective=sample_campaign_data['objective'],
            target_outcomes=sample_campaign_data['target_outcomes'],
            timeline=sample_campaign_data['timeline'],
            priority=sample_campaign_data['priority']
        )
        
        assert 'campaign_id' in result
        assert 'orchestration_plan' in result
        assert 'execution_status' in result
        assert 'estimated_timeline' in result
        assert 'success_probability' in result
        
        # Check campaign was stored
        campaign_id = result['campaign_id']
        assert campaign_id in orchestrator.active_campaigns
        
        stored_campaign = orchestrator.active_campaigns[campaign_id]
        assert stored_campaign['objective'] == sample_campaign_data['objective']
        assert stored_campaign['outcomes'] == sample_campaign_data['target_outcomes']
        assert stored_campaign['priority'] == sample_campaign_data['priority']
    
    @pytest.mark.asyncio
    async def test_campaign_with_constraints(self, orchestrator, sample_campaign_data, sample_constraints):
        """Test campaign orchestration with constraints"""
        result = await orchestrator.orchestrate_global_influence_campaign(
            campaign_objective=sample_campaign_data['objective'],
            target_outcomes=sample_campaign_data['target_outcomes'],
            timeline=sample_campaign_data['timeline'],
            priority=sample_campaign_data['priority'],
            constraints=sample_constraints
        )
        
        assert result is not None
        campaign_id = result['campaign_id']
        stored_campaign = orchestrator.active_campaigns[campaign_id]
        
        # Verify constraints were considered
        assert 'plan' in stored_campaign
        assert result['success_probability'] > 0
    
    def test_calculate_objective_complexity(self, orchestrator):
        """Test objective complexity calculation"""
        simple_objective = "Increase brand awareness"
        simple_outcomes = ["Improve social media presence"]
        
        complex_objective = "Establish global technology leadership across multiple domains"
        complex_outcomes = [
            "Become recognized AI thought leader",
            "Build strategic partnerships worldwide",
            "Influence industry standards and policies",
            "Create ecosystem of technology partners"
        ]
        
        simple_complexity = orchestrator._calculate_objective_complexity(simple_objective, simple_outcomes)
        complex_complexity = orchestrator._calculate_objective_complexity(complex_objective, complex_outcomes)
        
        assert complex_complexity > simple_complexity
        assert 0 <= simple_complexity <= 1
        assert 0 <= complex_complexity <= 1
    
    def test_determine_scope_level(self, orchestrator):
        """Test scope level determination"""
        global_objective = "Establish worldwide technology leadership"
        national_objective = "Become the leading AI company in the United States"
        local_objective = "Increase market share in Silicon Valley"
        
        assert orchestrator._determine_scope_level(global_objective) == "global"
        assert orchestrator._determine_scope_level(national_objective) == "national"
        assert orchestrator._determine_scope_level(local_objective) == "local"
    
    def test_identify_target_domains(self, orchestrator):
        """Test target domain identification"""
        tech_objective = "Lead AI innovation in healthcare technology"
        tech_outcomes = ["Develop AI medical diagnosis tools", "Partner with hospitals"]
        
        finance_objective = "Revolutionize banking with blockchain technology"
        finance_outcomes = ["Create digital payment platform", "Partner with banks"]
        
        tech_domains = orchestrator._identify_target_domains(tech_objective, tech_outcomes)
        finance_domains = orchestrator._identify_target_domains(finance_objective, finance_outcomes)
        
        assert "technology" in tech_domains
        assert "healthcare" in tech_domains
        assert "finance" in finance_domains
        assert "technology" in finance_domains
    
    def test_identify_stakeholder_types(self, orchestrator):
        """Test stakeholder type identification"""
        executive_objective = "Build relationships with Fortune 500 CEOs"
        executive_outcomes = ["Meet with top executives", "Join leadership forums"]
        
        investor_objective = "Secure Series B funding from top VCs"
        investor_outcomes = ["Present to venture capital firms", "Negotiate investment terms"]
        
        exec_types = orchestrator._identify_stakeholder_types(executive_objective, executive_outcomes)
        investor_types = orchestrator._identify_stakeholder_types(investor_objective, investor_outcomes)
        
        assert "executives" in exec_types
        assert "investors" in investor_types
    
    @pytest.mark.asyncio
    async def test_synchronize_influence_data(self, orchestrator):
        """Test data synchronization"""
        sync_results = await orchestrator.synchronize_influence_data()
        
        assert 'relationship_sync' in sync_results
        assert 'influence_sync' in sync_results
        assert 'partnership_sync' in sync_results
        assert 'network_sync' in sync_results
        assert 'timestamp' in sync_results
        
        # Check sync status was updated
        assert orchestrator.sync_status is not None
        assert 'last_sync' in orchestrator.sync_status
        assert 'sync_results' in orchestrator.sync_status
        assert 'sync_health' in orchestrator.sync_status
    
    @pytest.mark.asyncio
    async def test_get_influence_network_status(self, orchestrator):
        """Test network status retrieval"""
        # Add a test campaign first
        await orchestrator.orchestrate_global_influence_campaign(
            campaign_objective="Test campaign",
            target_outcomes=["Test outcome"],
            timeline=timedelta(days=30)
        )
        
        status = await orchestrator.get_influence_network_status()
        
        assert 'active_campaigns' in status
        assert 'network_health' in status
        assert 'influence_metrics' in status
        assert 'relationship_status' in status
        assert 'partnership_status' in status
        assert 'sync_status' in status
        assert 'performance_metrics' in status
        assert 'last_updated' in status
        
        assert status['active_campaigns'] >= 1  # At least our test campaign
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_campaigns(self, orchestrator):
        """Test handling multiple concurrent campaigns"""
        campaigns = []
        
        for i in range(3):
            result = await orchestrator.orchestrate_global_influence_campaign(
                campaign_objective=f"Test campaign {i+1}",
                target_outcomes=[f"Outcome {i+1}"],
                timeline=timedelta(days=30 + i*10),
                priority="medium"
            )
            campaigns.append(result)
        
        assert len(orchestrator.active_campaigns) == 3
        
        # Verify each campaign has unique ID
        campaign_ids = [c['campaign_id'] for c in campaigns]
        assert len(set(campaign_ids)) == 3
        
        # Verify all campaigns are stored correctly
        for campaign in campaigns:
            campaign_id = campaign['campaign_id']
            assert campaign_id in orchestrator.active_campaigns
    
    def test_orchestration_config(self, orchestrator):
        """Test orchestration configuration"""
        config = orchestrator.orchestration_config
        
        assert 'sync_interval' in config
        assert 'max_concurrent_campaigns' in config
        assert 'influence_threshold' in config
        assert 'relationship_priority_weights' in config
        
        weights = config['relationship_priority_weights']
        assert 'strategic_value' in weights
        assert 'influence_potential' in weights
        assert 'network_centrality' in weights
        assert 'accessibility' in weights
        
        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling in orchestration"""
        # Test with invalid parameters
        with pytest.raises(Exception):
            await orchestrator.orchestrate_global_influence_campaign(
                campaign_objective="",  # Empty objective
                target_outcomes=[],     # Empty outcomes
                timeline=timedelta(days=-1)  # Invalid timeline
            )
    
    @pytest.mark.asyncio
    async def test_campaign_lifecycle(self, orchestrator, sample_campaign_data):
        """Test complete campaign lifecycle"""
        # Create campaign
        result = await orchestrator.orchestrate_global_influence_campaign(
            campaign_objective=sample_campaign_data['objective'],
            target_outcomes=sample_campaign_data['target_outcomes'],
            timeline=sample_campaign_data['timeline'],
            priority=sample_campaign_data['priority']
        )
        
        campaign_id = result['campaign_id']
        
        # Verify initial state
        campaign = orchestrator.active_campaigns[campaign_id]
        assert campaign['objective'] == sample_campaign_data['objective']
        
        # Test status updates (would be done via API in real usage)
        campaign['status'] = 'active'
        campaign['last_updated'] = datetime.now()
        
        # Verify updated state
        updated_campaign = orchestrator.active_campaigns[campaign_id]
        assert updated_campaign['status'] == 'active'
        
        # Test completion
        campaign['status'] = 'completed'
        campaign['completion_date'] = datetime.now()
        
        assert orchestrator.active_campaigns[campaign_id]['status'] == 'completed'


class TestGlobalInfluenceModels:
    """Test suite for Global Influence Models"""
    
    def test_influence_campaign_creation(self):
        """Test influence campaign model creation"""
        campaign = InfluenceCampaign(
            objective="Test objective",
            target_outcomes=["Outcome 1", "Outcome 2"],
            timeline=timedelta(days=60),
            priority=CampaignPriority.HIGH
        )
        
        assert campaign.objective == "Test objective"
        assert len(campaign.target_outcomes) == 2
        assert campaign.timeline.days == 60
        assert campaign.priority == CampaignPriority.HIGH
        assert campaign.status == CampaignStatus.PLANNING
    
    def test_influence_target_creation(self):
        """Test influence target model creation"""
        target = InfluenceTarget(
            name="John Doe",
            title="CEO",
            organization="TechCorp",
            stakeholder_type=StakeholderType.EXECUTIVE,
            influence_score=0.8
        )
        
        assert target.name == "John Doe"
        assert target.title == "CEO"
        assert target.organization == "TechCorp"
        assert target.stakeholder_type == StakeholderType.EXECUTIVE
        assert target.influence_score == 0.8
    
    def test_influence_target_priority_calculation(self):
        """Test influence target priority score calculation"""
        target = InfluenceTarget(
            name="Jane Smith",
            strategic_value=0.9,
            influence_score=0.8,
            network_centrality=0.7,
            accessibility_score=0.6
        )
        
        weights = {
            'strategic_value': 0.4,
            'influence_potential': 0.3,
            'network_centrality': 0.2,
            'accessibility': 0.1
        }
        
        priority_score = target.calculate_priority_score(weights)
        
        expected_score = (0.9 * 0.4) + (0.8 * 0.3) + (0.7 * 0.2) + (0.6 * 0.1)
        assert abs(priority_score - expected_score) < 0.01
    
    def test_influence_network_density(self):
        """Test influence network density calculation"""
        network = InfluenceNetwork(
            name="Test Network",
            targets=[
                InfluenceTarget(name="Target 1"),
                InfluenceTarget(name="Target 2"),
                InfluenceTarget(name="Target 3")
            ]
        )
        
        # Add some connections
        target_ids = [t.id for t in network.targets]
        network.connections = {
            target_ids[0]: [target_ids[1]],
            target_ids[1]: [target_ids[0], target_ids[2]],
            target_ids[2]: [target_ids[1]]
        }
        
        density = network.get_network_density()
        
        # 4 total connections, 6 possible connections (3*2)
        expected_density = 4 / 6
        assert abs(density - expected_density) < 0.01
    
    def test_influence_network_central_nodes(self):
        """Test getting central nodes from network"""
        targets = [
            InfluenceTarget(name="Target 1", network_centrality=0.9),
            InfluenceTarget(name="Target 2", network_centrality=0.7),
            InfluenceTarget(name="Target 3", network_centrality=0.8),
            InfluenceTarget(name="Target 4", network_centrality=0.6)
        ]
        
        network = InfluenceNetwork(
            name="Test Network",
            targets=targets
        )
        
        central_nodes = network.get_central_nodes(top_n=2)
        
        assert len(central_nodes) == 2
        assert central_nodes[0].network_centrality == 0.9
        assert central_nodes[1].network_centrality == 0.8


if __name__ == "__main__":
    pytest.main([__file__])