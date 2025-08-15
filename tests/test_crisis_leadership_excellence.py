"""
Tests for Crisis Leadership Excellence System

Comprehensive tests for the complete crisis leadership excellence system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.crisis_leadership_excellence import (
    CrisisLeadershipExcellence,
    CrisisType,
    CrisisSeverity,
    CrisisStatus,
    CrisisContext,
    CrisisResponse
)


class TestCrisisLeadershipExcellence:
    """Test suite for Crisis Leadership Excellence system"""
    
    @pytest.fixture
    def crisis_system(self):
        """Create crisis leadership system instance"""
        return CrisisLeadershipExcellence()
    
    @pytest.fixture
    def sample_crisis_signals(self):
        """Sample crisis signals for testing"""
        return [
            {
                'type': 'system_alert',
                'severity': 'high',
                'message': 'Database connection failure detected',
                'timestamp': datetime.now(),
                'source': 'monitoring_system',
                'metadata': {'affected_services': ['api', 'web']}
            },
            {
                'type': 'user_complaint',
                'severity': 'medium',
                'message': 'Users reporting login issues',
                'timestamp': datetime.now(),
                'source': 'support_system',
                'metadata': {'complaint_count': 15}
            }
        ]
    
    @pytest.fixture
    def sample_crisis_scenarios(self):
        """Sample crisis scenarios for validation testing"""
        return [
            {
                'id': 'scenario_1',
                'type': 'system_outage',
                'signals': [
                    {
                        'type': 'system_alert',
                        'severity': 'critical',
                        'message': 'Complete system outage',
                        'timestamp': datetime.now()
                    }
                ],
                'expected_outcomes': ['system_restored', 'stakeholders_notified']
            },
            {
                'id': 'scenario_2',
                'type': 'security_breach',
                'signals': [
                    {
                        'type': 'security_incident',
                        'severity': 'high',
                        'message': 'Unauthorized access detected',
                        'timestamp': datetime.now()
                    }
                ],
                'expected_outcomes': ['breach_contained', 'security_enhanced']
            }
        ]
    
    @pytest.mark.asyncio
    async def test_handle_crisis_basic(self, crisis_system, sample_crisis_signals):
        """Test basic crisis handling functionality"""
        
        # Mock the detection engine
        with patch.object(crisis_system.crisis_detector, 'detect_crisis') as mock_detect:
            mock_detect.return_value = {'crisis_detected': True, 'confidence': 0.9}
            
            with patch.object(crisis_system.crisis_detector, 'classify_crisis') as mock_classify:
                mock_classify.return_value = {
                    'type': 'system_outage',
                    'severity': 'high',
                    'description': 'System outage detected'
                }
                
                with patch.object(crisis_system.crisis_detector, 'assess_impact') as mock_impact:
                    mock_impact.return_value = {
                        'affected_systems': ['api', 'database'],
                        'stakeholders': ['customers', 'employees'],
                        'impact_metrics': {'severity_score': 0.8}
                    }
                    
                    # Mock other engines
                    crisis_system.decision_engine.get_decision_options = AsyncMock(return_value=[
                        {'name': 'immediate_response', 'priority': 'high'}
                    ])
                    crisis_system.info_synthesizer.synthesize_crisis_information = AsyncMock(return_value={
                        'synthesized_info': 'Crisis information processed'
                    })
                    crisis_system.risk_analyzer.analyze_response_options = AsyncMock(return_value={
                        'recommended_actions': [{'name': 'restore_service', 'priority': 'critical'}],
                        'contingency_options': [],
                        'risk_mitigation_steps': [],
                        'success_metrics': {'restoration_time': 30},
                        'recommended_timeline': []
                    })
                    
                    # Mock team and resource systems
                    crisis_system.team_former.form_crisis_team = AsyncMock(return_value={
                        'team_members': ['engineer_1', 'manager_1'],
                        'team_structure': 'incident_response',
                        'team_requirements': []
                    })
                    crisis_system.role_assigner.assign_crisis_roles = AsyncMock(return_value={
                        'engineer_1': 'technical_lead',
                        'manager_1': 'incident_commander'
                    })
                    crisis_system.resource_assessor.assess_crisis_resources = AsyncMock(return_value={
                        'available_resources': {'engineers': 5, 'servers': 10}
                    })
                    crisis_system.resource_allocator.optimize_crisis_allocation = AsyncMock(return_value={
                        'allocation_plan': {'engineers': 3, 'servers': 5}
                    })
                    crisis_system.external_coordinator.coordinate_external_support = AsyncMock(return_value={
                        'external_resources': []
                    })
                    
                    # Mock communication systems
                    crisis_system.stakeholder_notifier.notify_crisis_stakeholders = AsyncMock(return_value={
                        'messages': [{'recipient': 'customers', 'message': 'Service disruption notice'}]
                    })
                    crisis_system.message_coordinator.coo