"""
Tests for Board Strategic Integration

Tests the integration between board executive mastery and strategic planning systems.
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.board_strategic_integration import (
    BoardStrategicIntegration,
    BoardStrategicAlignment,
    BoardFeedbackIntegration,
    BoardApprovalTracking
)
from scrollintel.models.board_dynamics_models import Board, BoardMember
from scrollintel.models.strategic_planning_models import TechnologyVision, StrategicRoadmap
from scrollintel.engines.board_dynamics_engine import (
    Background, Priority, InfluenceLevel, CommunicationStyle, DecisionPattern
)


class TestBoardStrategicIntegration:
    """Test board strategic integration functionality"""
    
    @pytest.fixture
    def integration_system(self):
        """Create integration system for testing"""
        return BoardStrategicIntegration()
    
    @pytest.fixture
    def mock_board(self):
        """Create mock board for testing"""
        member1 = BoardMember(
            id="member_1",
            name="Alice Johnson",
            background=Background(
                industry_experience=["technology", "startup"],
                functional_expertise=["strategy", "product"],
                education=["MBA", "Computer Science"],
                previous_roles=["CEO", "VP Product"],
                years_experience=15
            ),
            expertise_areas=["AI", "product strategy"],
            influence_level=InfluenceLevel.HIGH,
            communication_style=CommunicationStyle.VISIONARY,
            decision_making_pattern=DecisionPattern.INTUITIVE,
            priorities=[
                Priority(
                    area="AI Innovation",
                    importance=0.9,
                    description="Lead in AI technology",
                    timeline="2-3 years"
                ),
                Priority(
                    area="Market Expansion",
                    importance=0.7,
                    description="Expand to new markets",
                    timeline="1-2 years"
                )
            ]
        )
        
        member2 = BoardMember(
            id="member_2",
            name="Bob Smith",
            background=Background(
                industry_experience=["finance", "consulting"],
                functional_expertise=["finance", "risk management"],
                education=["MBA", "Finance"],
                previous_roles=["CFO", "Partner"],
                years_experience=20
            ),
            expertise_areas=["financial planning", "risk assessment"],
            influence_level=InfluenceLevel.MEDIUM,
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_making_pattern=DecisionPattern.DATA_DRIVEN,
            priorities=[
                Priority(
                    area="Financial Performance",
                    importance=0.8,
                    description="Improve financial metrics",
                    timeline="1 year"
                ),
                Priority(
                    area="Risk Management",
                    importance=0.6,
                    description="Minimize operational risks",
                    timeline="ongoing"
                )
            ]
        )
        
        return Board(
            id="board_1",
            name="Test Board",
            members=[member1, member2],
            committees=["Audit", "Compensation"],
            governance_structure={"type": "traditional"},
            meeting_schedule=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.fixture
    def mock_vision(self):
        """Create mock technology vision for testing"""
        return TechnologyVision(
            title="AI-First Technology Strategy",
            description="Transform company through AI technology leadership",
            time_horizon=5,
            key_technologies=["artificial_intelligence", "machine_learning", "automation"],
            market_assumptions={
                "ai_market_growth": "30% annually",
                "competitive_landscape": "intensifying",
                "regulatory_environment": "evolving"
            },
            success_criteria=[
                "Market leadership in AI solutions",
                "50% revenue from AI products",
                "Industry recognition as AI innovator"
            ],
            stakeholders=["CTO", "CEO", "Product Team"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_create_board_aligned_strategic_plan(self, integration_system, mock_board, mock_vision):
        """Test creating board-aligned strategic plan"""
        with patch.object(integration_system.board_dynamics, 'analyze_board_composition', new_callable=AsyncMock) as mock_analyze:
            with patch.object(integration_system.board_dynamics, 'map_power_structures', new_callable=AsyncMock) as mock_map:
                with patch.object(integration_system.strategic_planner, 'create_longterm_roadmap', new_callable=AsyncMock) as mock_create:
                    
                    # Setup mocks
                    mock_analyze.return_value = {"composition_score": 0.8}
                    mock_map.return_value = Mock()
                    mock_create.return_value = Mock(
                        id="roadmap_1",
                        name="Test Roadmap",
                        description="Test Description",
                        vision=mock_vision,
                        time_horizon=5,
                        milestones=[],
                        technology_bets=[],
                        risk_assessments=[],
                        success_metrics=[],
                        stakeholders=["CTO"],
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    # Test board-aligned strategic plan creation
                    roadmap = await integration_system.create_board_aligned_strategic_plan(
                        mock_board, mock_vision, 5
                    )
                    
                    # Verify results
                    assert roadmap is not None
                    assert mock_analyze.called
                    assert mock_map.called
                    assert mock_create.called
    
    def test_extract_board_priorities(self, integration_system, mock_board):
        """Test extracting and consolidating board priorities"""
        priorities = integration_system._extract_board_priorities(mock_board.members)
        
        # Verify priority extraction
        assert len(priorities) > 0
        assert all('area' in p for p in priorities)
        assert all('importance' in p for p in priorities)
        assert all('member_support' in p for p in priorities)
        
        # Verify priorities are sorted by importance and support
        for i in range(len(priorities) - 1):
            current_weight = priorities[i]['importance'] * priorities[i]['member_support']
            next_weight = priorities[i + 1]['importance'] * priorities[i + 1]['member_support']
            assert current_weight >= next_weight
    
    def test_assess_board_risk_tolerance(self, integration_system, mock_board):
        """Test assessing board risk tolerance"""
        risk_tolerance = integration_system._assess_board_risk_tolerance(mock_board.members)
        
        # Verify risk tolerance calculation
        assert 0.0 <= risk_tolerance <= 1.0
        assert isinstance(risk_tolerance, float)
    
    @pytest.mark.asyncio
    async def test_integrate_board_feedback(self, integration_system):
        """Test integrating board feedback into strategic planning"""
        # Create mock feedback
        feedback_items = [
            BoardFeedbackIntegration(
                feedback_id="feedback_1",
                board_member_id="member_1",
                strategic_element="AI Investment",
                feedback_type="timeline_adjustment",
                feedback_content="Accelerate AI development timeline",
                impact_assessment=0.8,
                integration_status="pending",
                created_at=datetime.now()
            ),
            BoardFeedbackIntegration(
                feedback_id="feedback_2",
                board_member_id="member_2",
                strategic_element="Risk Management",
                feedback_type="risk_mitigation",
                feedback_content="Add additional risk controls",
                impact_assessment=0.6,
                integration_status="pending",
                created_at=datetime.now()
            )
        ]
        
        # Create mock roadmap
        mock_roadmap = Mock()
        mock_roadmap.id = "roadmap_1"
        mock_roadmap.milestones = []
        mock_roadmap.risk_assessments = []
        mock_roadmap.updated_at = datetime.now()
        
        # Test feedback integration
        updated_roadmap = await integration_system.integrate_board_feedback(
            feedback_items, mock_roadmap
        )
        
        # Verify integration
        assert updated_roadmap is not None
        assert all(f.integration_status == "integrated" for f in feedback_items)
    
    @pytest.mark.asyncio
    async def test_track_board_approval(self, integration_system, mock_board):
        """Test tracking board approval for strategic initiatives"""
        voting_record = {
            "member_1": "approve",
            "member_2": "conditional:need more financial analysis"
        }
        
        # Test approval tracking
        tracking = await integration_system.track_board_approval(
            "initiative_1", mock_board, voting_record
        )
        
        # Verify tracking results
        assert tracking.initiative_id == "initiative_1"
        assert tracking.board_id == mock_board.id
        assert tracking.approval_status in ["approved", "conditional_approval", "rejected"]
        assert tracking.voting_record == voting_record
        assert len(tracking.approval_conditions) >= 0
        assert tracking.next_review_date > date.today()
    
    @pytest.mark.asyncio
    async def test_generate_board_strategic_adjustment(self, integration_system, mock_board):
        """Test generating strategic adjustments based on board input"""
        # Create mock roadmap
        mock_roadmap = Mock()
        mock_roadmap.id = "roadmap_1"
        mock_roadmap.name = "Test Roadmap"
        
        # Create mock market changes
        market_changes = [
            {
                "type": "technology_disruption",
                "description": "New AI breakthrough announced by competitor",
                "impact_level": 0.8,
                "urgency": "high"
            },
            {
                "type": "market_shift",
                "description": "Customer preferences shifting to AI-first solutions",
                "impact_level": 0.7,
                "urgency": "medium"
            }
        ]
        
        # Test strategic adjustment generation
        pivot = await integration_system.generate_board_strategic_adjustment(
            mock_board, mock_roadmap, market_changes
        )
        
        # Verify pivot generation
        assert pivot.id is not None
        assert pivot.name == "Board-Driven Strategic Adjustment"
        assert pivot.pivot_type == "market_response"
        assert pivot.timeline == 90
        assert len(pivot.trigger_events) == len(market_changes)
        assert pivot.board_approval_required is True
    
    @pytest.mark.asyncio
    async def test_assess_strategic_alignment(self, integration_system, mock_board):
        """Test assessing strategic alignment between board and roadmap"""
        # Create mock roadmap
        mock_roadmap = Mock()
        mock_roadmap.id = "roadmap_1"
        mock_roadmap.technology_bets = []
        mock_roadmap.market_assumptions = {
            "ai_innovation": "High priority for competitive advantage",
            "market_expansion": "Essential for growth"
        }
        
        # Test alignment assessment
        alignment = await integration_system._assess_strategic_alignment(
            mock_board, mock_roadmap
        )
        
        # Verify alignment assessment
        assert alignment.board_id == mock_board.id
        assert alignment.strategic_plan_id == mock_roadmap.id
        assert 0.0 <= alignment.alignment_score <= 1.0
        assert isinstance(alignment.priority_matches, list)
        assert isinstance(alignment.concern_areas, list)
        assert isinstance(alignment.recommendations, list)
        assert alignment.last_updated is not None


class TestBoardStrategicIntegrationAPI:
    """Test board strategic integration API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from scrollintel.api.routes.board_strategic_integration_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_create_aligned_plan_endpoint(self, client):
        """Test create board-aligned strategic plan endpoint"""
        with patch('scrollintel.api.routes.board_strategic_integration_routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user"}
            
            board_data = {
                "id": "board_1",
                "name": "Test Board",
                "members": [],
                "committees": [],
                "governance_structure": {},
                "meeting_schedule": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            vision_data = {
                "title": "Test Vision",
                "description": "Test Description",
                "time_horizon": 5,
                "key_technologies": ["AI"],
                "market_assumptions": {"growth": "high"},
                "success_criteria": ["leadership"],
                "stakeholders": ["CTO"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            with patch('scrollintel.core.board_strategic_integration.BoardStrategicIntegration') as mock_integration:
                mock_instance = Mock()
                mock_instance.create_board_aligned_strategic_plan = AsyncMock()
                mock_instance.create_board_aligned_strategic_plan.return_value = Mock(
                    id="roadmap_1",
                    name="Test Roadmap",
                    description="Test Description",
                    time_horizon=5,
                    milestones=[],
                    technology_bets=[],
                    stakeholders=["CTO"]
                )
                mock_integration.return_value = mock_instance
                
                response = client.post(
                    "/api/v1/board-strategic/create-aligned-plan",
                    json={
                        "board_data": board_data,
                        "vision_data": vision_data,
                        "horizon": 5
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "roadmap" in data
    
    def test_integrate_feedback_endpoint(self, client):
        """Test integrate board feedback endpoint"""
        with patch('scrollintel.api.routes.board_strategic_integration_routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user"}
            
            feedback_items = [
                {
                    "board_member_id": "member_1",
                    "strategic_element": "AI Investment",
                    "feedback_type": "timeline_adjustment",
                    "feedback_content": "Accelerate timeline",
                    "impact_assessment": 0.8
                }
            ]
            
            with patch('scrollintel.core.board_strategic_integration.BoardStrategicIntegration') as mock_integration:
                with patch('scrollintel.api.routes.board_strategic_integration_routes._get_roadmap_by_id') as mock_get_roadmap:
                    mock_instance = Mock()
                    mock_instance.integrate_board_feedback = AsyncMock()
                    mock_instance.integrate_board_feedback.return_value = Mock(
                        id="roadmap_1",
                        name="Updated Roadmap",
                        updated_at=datetime.now()
                    )
                    mock_integration.return_value = mock_instance
                    mock_get_roadmap.return_value = Mock()
                    
                    response = client.post(
                        "/api/v1/board-strategic/integrate-feedback",
                        json={
                            "roadmap_id": "roadmap_1",
                            "feedback_items": feedback_items
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert "integrated_feedback" in data
    
    def test_track_approval_endpoint(self, client):
        """Test track board approval endpoint"""
        with patch('scrollintel.api.routes.board_strategic_integration_routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user"}
            
            board_data = {
                "id": "board_1",
                "name": "Test Board",
                "members": [],
                "committees": [],
                "governance_structure": {},
                "meeting_schedule": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            voting_record = {
                "member_1": "approve",
                "member_2": "reject"
            }
            
            with patch('scrollintel.core.board_strategic_integration.BoardStrategicIntegration') as mock_integration:
                mock_instance = Mock()
                mock_instance.track_board_approval = AsyncMock()
                mock_instance.track_board_approval.return_value = BoardApprovalTracking(
                    initiative_id="initiative_1",
                    board_id="board_1",
                    approval_status="rejected",
                    voting_record=voting_record,
                    approval_conditions=[],
                    next_review_date=date.today() + timedelta(days=90),
                    approval_history=[]
                )
                mock_integration.return_value = mock_instance
                
                response = client.post(
                    "/api/v1/board-strategic/track-approval",
                    json={
                        "initiative_id": "initiative_1",
                        "board_data": board_data,
                        "voting_record": voting_record
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "approval_tracking" in data


if __name__ == "__main__":
    pytest.main([__file__])