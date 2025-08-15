"""
Tests for Board Communication Integration

Tests the integration between board executive mastery and communication systems.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.board_communication_integration import (
    BoardCommunicationIntegration,
    BoardContextualMessage,
    CommunicationIntegrationConfig,
    BoardResponseGeneration,
    CommunicationChannel
)
from scrollintel.models.board_dynamics_models import Board, BoardMember
from scrollintel.models.executive_communication_models import Message, MessageType, ExecutiveLevel
from scrollintel.engines.board_dynamics_engine import (
    Background, Priority, InfluenceLevel, CommunicationStyle, DecisionPattern
)


class TestBoardCommunicationIntegration:
    """Test board communication integration functionality"""
    
    @pytest.fixture
    def integration_system(self):
        """Create integration system for testing"""
        return BoardCommunicationIntegration()
    
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
    def mock_message(self):
        """Create mock message for testing"""
        return Message(
            id="msg_1",
            content="Strategic update on AI initiative progress and financial implications for board review",
            sender="cto",
            recipients=["board"],
            message_type=MessageType.STRATEGIC_UPDATE,
            urgency="medium",
            key_points=[
                "AI initiative ahead of schedule",
                "Budget utilization at 75%",
                "Market response positive"
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.fixture
    def mock_config(self):
        """Create mock integration config for testing"""
        return CommunicationIntegrationConfig(
            board_id="board_1",
            enabled_channels=[CommunicationChannel.EMAIL, CommunicationChannel.BOARD_MEETING],
            auto_adaptation=True,
            context_awareness_level="high",
            response_generation_mode="automatic",
            escalation_rules={"high_urgency": "immediate_notification"}
        )
    
    @pytest.mark.asyncio
    async def test_create_board_contextual_communication(
        self, integration_system, mock_message, mock_board, mock_config
    ):
        """Test creating board contextual communication"""
        with patch.object(integration_system.board_dynamics, 'analyze_board_composition', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {"composition_score": 0.8}
            
            # Test contextual communication creation
            contextual_message = await integration_system.create_board_contextual_communication(
                mock_message, mock_board, CommunicationChannel.EMAIL, mock_config
            )
            
            # Verify results
            assert contextual_message is not None
            assert contextual_message.message_id is not None
            assert contextual_message.channel == CommunicationChannel.EMAIL
            assert contextual_message.board_relevance_score >= 0.0
            assert contextual_message.board_relevance_score <= 1.0
            assert len(contextual_message.adapted_versions) == len(mock_board.members)
            assert len(contextual_message.board_member_profiles) == len(mock_board.members)
    
    @pytest.mark.asyncio
    async def test_generate_board_appropriate_response(
        self, integration_system, mock_message, mock_board
    ):
        """Test generating board-appropriate response"""
        board_member = mock_board.members[0]
        response_context = {"meeting_type": "board_meeting", "urgency": "medium"}
        
        # Test response generation
        response_generation = await integration_system.generate_board_appropriate_response(
            mock_message, board_member, response_context
        )
        
        # Verify response generation
        assert response_generation is not None
        assert response_generation.response_id is not None
        assert response_generation.board_member_id == board_member.id
        assert response_generation.generated_response is not None
        assert len(response_generation.generated_response) > 0
        assert response_generation.response_tone is not None
        assert 0.0 <= response_generation.board_appropriateness_score <= 1.0
        assert isinstance(response_generation.key_messages, list)
        assert isinstance(response_generation.follow_up_actions, list)
    
    def test_create_member_communication_profiles(self, integration_system, mock_board):
        """Test creating member communication profiles"""
        profiles = asyncio.run(
            integration_system._create_member_communication_profiles(mock_board.members)
        )
        
        # Verify profiles
        assert len(profiles) == len(mock_board.members)
        for profile in profiles:
            assert 'member_id' in profile
            assert 'name' in profile
            assert 'communication_style' in profile
            assert 'decision_pattern' in profile
            assert 'influence_level' in profile
            assert 'expertise_areas' in profile
            assert 'preferred_detail_level' in profile
            assert 'attention_span' in profile
            assert 'response_style' in profile
            assert 'key_concerns' in profile
    
    @pytest.mark.asyncio
    async def test_create_executive_audience(self, integration_system, mock_board):
        """Test creating executive audience from board member"""
        board_member = mock_board.members[0]
        
        audience = await integration_system._create_executive_audience(board_member)
        
        # Verify audience creation
        assert audience is not None
        assert audience.id == f"audience_{board_member.id}"
        assert audience.name == board_member.name
        assert audience.executive_level in [ExecutiveLevel.BOARD_MEMBER, ExecutiveLevel.CEO, ExecutiveLevel.CTO, ExecutiveLevel.CFO, ExecutiveLevel.BOARD_CHAIR]
        assert audience.expertise_areas == board_member.expertise_areas
        assert len(audience.priorities) > 0
    
    def test_determine_detail_preference(self, integration_system, mock_board):
        """Test determining detail preference"""
        analytical_member = mock_board.members[1]  # Bob Smith with analytical style
        visionary_member = mock_board.members[0]   # Alice Johnson with visionary style
        
        analytical_preference = integration_system._determine_detail_preference(analytical_member)
        visionary_preference = integration_system._determine_detail_preference(visionary_member)
        
        # Verify preferences
        assert analytical_preference == "high"
        assert visionary_preference in ["low", "medium"]
    
    def test_estimate_attention_span(self, integration_system, mock_board):
        """Test estimating attention span"""
        data_driven_member = mock_board.members[1]  # Bob Smith with data-driven pattern
        intuitive_member = mock_board.members[0]    # Alice Johnson with intuitive pattern
        
        data_driven_span = integration_system._estimate_attention_span(data_driven_member)
        intuitive_span = integration_system._estimate_attention_span(intuitive_member)
        
        # Verify attention spans
        assert isinstance(data_driven_span, int)
        assert isinstance(intuitive_span, int)
        assert data_driven_span > 0
        assert intuitive_span > 0
        # Data-driven members typically have longer attention spans
        assert data_driven_span >= intuitive_span
    
    @pytest.mark.asyncio
    async def test_calculate_board_relevance(self, integration_system, mock_message, mock_board):
        """Test calculating board relevance score"""
        relevance_score = await integration_system._calculate_board_relevance(mock_message, mock_board)
        
        # Verify relevance calculation
        assert 0.0 <= relevance_score <= 1.0
        assert isinstance(relevance_score, float)
    
    @pytest.mark.asyncio
    async def test_determine_urgency_level(self, integration_system, mock_message):
        """Test determining urgency level"""
        board_context = {
            'relevant_priorities': [{'priority': 'AI Innovation', 'importance': 0.9}],
            'key_stakeholders': ['member_1', 'member_2']
        }
        
        urgency_level = await integration_system._determine_urgency_level(mock_message, board_context)
        
        # Verify urgency determination
        assert urgency_level in ["low", "medium", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_generate_response_content(self, integration_system, mock_message, mock_board):
        """Test generating response content"""
        board_member = mock_board.members[0]
        context = {"meeting_type": "board_meeting"}
        
        response_content = await integration_system._generate_response_content(
            mock_message, board_member, context
        )
        
        # Verify response content
        assert isinstance(response_content, str)
        assert len(response_content) > 0
        # Should contain professional language
        assert any(word in response_content.lower() for word in ['thank', 'perspective', 'recommend', 'consider'])
    
    @pytest.mark.asyncio
    async def test_determine_response_tone(self, integration_system, mock_message, mock_board):
        """Test determining response tone"""
        analytical_member = mock_board.members[1]  # Bob Smith with analytical style
        visionary_member = mock_board.members[0]   # Alice Johnson with visionary style
        
        analytical_tone = await integration_system._determine_response_tone(analytical_member, mock_message)
        visionary_tone = await integration_system._determine_response_tone(visionary_member, mock_message)
        
        # Verify tone determination
        assert analytical_tone == "professional_analytical"
        assert visionary_tone == "inspiring_strategic"
    
    @pytest.mark.asyncio
    async def test_calculate_appropriateness_score(self, integration_system, mock_board, mock_message):
        """Test calculating board appropriateness score"""
        board_member = mock_board.members[0]
        response_content = "I recommend we consider this strategic initiative carefully and analyze the data."
        
        score = await integration_system._calculate_appropriateness_score(
            response_content, board_member, mock_message
        )
        
        # Verify appropriateness score
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_build_board_context_awareness(self, integration_system, mock_board):
        """Test building board context awareness"""
        # Create mock communication history
        communication_history = [
            Message(
                id=f"msg_{i}",
                content=f"Message {i} about strategic planning and AI innovation",
                sender="system",
                recipients=["board"],
                message_type=MessageType.STRATEGIC_UPDATE,
                urgency="medium",
                key_points=[f"Point {i}"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(5)
        ]
        
        # Test context awareness building
        context_awareness = await integration_system.build_board_context_awareness(
            communication_history, mock_board, 30
        )
        
        # Verify context awareness
        assert context_awareness is not None
        assert 'board_id' in context_awareness
        assert 'analysis_period' in context_awareness
        assert 'communication_patterns' in context_awareness
        assert 'recurring_themes' in context_awareness
        assert 'member_engagement' in context_awareness
        assert 'effectiveness_metrics' in context_awareness
        assert 'context_insights' in context_awareness
        assert context_awareness['board_id'] == mock_board.id
    
    @pytest.mark.asyncio
    async def test_analyze_communication_patterns(self, integration_system, mock_board):
        """Test analyzing communication patterns"""
        # Create mock communication history
        communication_history = [
            Mock(channel="email", urgency="medium"),
            Mock(channel="board_meeting", urgency="high"),
            Mock(channel="email", urgency="low")
        ]
        
        patterns = await integration_system._analyze_communication_patterns(
            communication_history, mock_board
        )
        
        # Verify pattern analysis
        assert 'message_frequency' in patterns
        assert 'channel_distribution' in patterns
        assert 'urgency_distribution' in patterns
        assert 'response_times' in patterns
        assert 'member_participation' in patterns
        assert patterns['message_frequency'] == len(communication_history)
    
    @pytest.mark.asyncio
    async def test_identify_recurring_themes(self, integration_system):
        """Test identifying recurring themes"""
        # Create mock messages with themes
        communication_history = [
            Mock(id="msg_1", content="Strategic planning for AI innovation and growth"),
            Mock(id="msg_2", content="Financial performance and risk management update"),
            Mock(id="msg_3", content="Market expansion strategy and innovation roadmap"),
            Mock(id="msg_4", content="Risk assessment for strategic initiatives"),
            Mock(id="msg_5", content="Innovation pipeline and financial projections")
        ]
        
        themes = await integration_system._identify_recurring_themes(communication_history)
        
        # Verify theme identification
        assert isinstance(themes, list)
        assert len(themes) > 0
        for theme in themes:
            assert 'theme' in theme
            assert 'frequency' in theme
            assert 'message_ids' in theme
            assert theme['frequency'] > 0
    
    @pytest.mark.asyncio
    async def test_track_member_engagement(self, integration_system, mock_board):
        """Test tracking member engagement"""
        communication_history = [Mock() for _ in range(10)]  # 10 mock messages
        
        engagement = await integration_system._track_member_engagement(
            communication_history, mock_board
        )
        
        # Verify engagement tracking
        assert isinstance(engagement, dict)
        assert len(engagement) == len(mock_board.members)
        
        for member_id, metrics in engagement.items():
            assert 'name' in metrics
            assert 'messages_initiated' in metrics
            assert 'responses_provided' in metrics
            assert 'avg_response_time' in metrics
            assert 'engagement_score' in metrics
            assert 0.0 <= metrics['engagement_score'] <= 1.0


class TestBoardCommunicationIntegrationAPI:
    """Test board communication integration API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from scrollintel.api.routes.board_communication_integration_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_create_contextual_communication_endpoint(self, client):
        """Test create board contextual communication endpoint"""
        with patch('scrollintel.api.routes.board_communication_integration_routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user"}
            
            message_data = {
                "id": "msg_1",
                "content": "Test message content",
                "sender": "system",
                "recipients": ["board"],
                "message_type": "strategic_update",
                "urgency": "medium",
                "key_points": ["Point 1", "Point 2"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
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
            
            config_data = {
                "board_id": "board_1",
                "enabled_channels": ["email"],
                "auto_adaptation": True,
                "context_awareness_level": "high",
                "response_generation_mode": "automatic",
                "escalation_rules": {}
            }
            
            with patch('scrollintel.core.board_communication_integration.BoardCommunicationIntegration') as mock_integration:
                mock_instance = Mock()
                mock_instance.create_board_contextual_communication = AsyncMock()
                mock_instance.create_board_contextual_communication.return_value = Mock(
                    message_id="ctx_msg_1",
                    channel=CommunicationChannel.EMAIL,
                    urgency_level="medium",
                    board_relevance_score=0.8,
                    adapted_versions={"member_1": Mock()},
                    board_member_profiles=[{"member_id": "member_1"}]
                )
                mock_integration.return_value = mock_instance
                
                response = client.post(
                    "/api/v1/board-communication/create-contextual-communication",
                    json={
                        "message_data": message_data,
                        "board_data": board_data,
                        "channel": "email",
                        "config_data": config_data
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "contextual_message" in data
    
    def test_generate_board_response_endpoint(self, client):
        """Test generate board-appropriate response endpoint"""
        with patch('scrollintel.api.routes.board_communication_integration_routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user"}
            
            original_message_data = {
                "id": "msg_1",
                "content": "Original message content",
                "sender": "system",
                "recipients": ["board"],
                "message_type": "strategic_update",
                "urgency": "medium",
                "key_points": ["Point 1"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            board_member_data = {
                "id": "member_1",
                "name": "John Doe",
                "background": {
                    "industry_experience": ["technology"],
                    "functional_expertise": ["strategy"],
                    "education": ["MBA"],
                    "previous_roles": ["CEO"],
                    "years_experience": 15
                },
                "expertise_areas": ["strategy"],
                "influence_level": "high",
                "communication_style": "analytical",
                "decision_making_pattern": "data_driven",
                "priorities": []
            }
            
            response_context = {"meeting_type": "board_meeting"}
            
            with patch('scrollintel.core.board_communication_integration.BoardCommunicationIntegration') as mock_integration:
                mock_instance = Mock()
                mock_instance.generate_board_appropriate_response = AsyncMock()
                mock_instance.generate_board_appropriate_response.return_value = BoardResponseGeneration(
                    response_id="resp_1",
                    original_message_id="msg_1",
                    board_member_id="member_1",
                    generated_response="Generated response content",
                    response_tone="professional_analytical",
                    board_appropriateness_score=0.85,
                    key_messages=["Key message 1", "Key message 2"],
                    follow_up_actions=["Action 1", "Action 2"],
                    created_at=datetime.now()
                )
                mock_integration.return_value = mock_instance
                
                response = client.post(
                    "/api/v1/board-communication/generate-board-response",
                    json={
                        "original_message_data": original_message_data,
                        "board_member_data": board_member_data,
                        "response_context": response_context
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "response" in data
                assert data["response"]["board_appropriateness_score"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])