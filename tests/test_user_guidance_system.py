"""
Tests for User Guidance and Support System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.user_guidance_system import UserGuidanceSystem
from scrollintel.models.user_guidance_models import (
    GuidanceContext, ErrorExplanation, ProactiveGuidance,
    SupportTicket, UserBehaviorPattern, GuidanceType,
    SeverityLevel, TicketStatus
)

class TestUserGuidanceSystem:
    """Test cases for UserGuidanceSystem"""
    
    @pytest.fixture
    def guidance_system(self):
        """Create a UserGuidanceSystem instance for testing"""
        return UserGuidanceSystem()
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample guidance context"""
        return GuidanceContext(
            user_id="test_user_123",
            session_id="session_456",
            current_page="/dashboard",
            user_action="click_button",
            system_state={"load": 0.5, "errors": 0}
        )
    
    @pytest.mark.asyncio
    async def test_provide_contextual_help_success(self, guidance_system, sample_context):
        """Test successful contextual help provision"""
        # Mock the internal methods
        with patch.object(guidance_system, '_analyze_context_for_help') as mock_analyze, \
             patch.object(guidance_system, '_generate_contextual_guidance') as mock_generate, \
             patch.object(guidance_system, '_track_help_provision') as mock_track:
            
            mock_analyze.return_value = {"user_level": "intermediate"}
            mock_generate.return_value = {
                "type": GuidanceType.CONTEXTUAL_HELP,
                "title": "Dashboard Help",
                "content": "Here's how to use the dashboard...",
                "confidence_score": 0.85
            }
            mock_track.return_value = None
            
            result = await guidance_system.provide_contextual_help(sample_context)
            
            assert result is not None
            assert result["type"] == GuidanceType.CONTEXTUAL_HELP
            assert result["title"] == "Dashboard Help"
            assert result["confidence_score"] == 0.85
            
            mock_analyze.assert_called_once_with(sample_context)
            mock_generate.assert_called_once()
            mock_track.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_provide_contextual_help_with_cache(self, guidance_system, sample_context):
        """Test contextual help with cached content"""
        # Set up cache
        cache_key = await guidance_system._generate_help_cache_key(sample_context)
        guidance_system.help_cache[cache_key] = {
            'guidance': {"title": "Cached Help", "confidence_score": 0.9},
            'timestamp': datetime.utcnow(),
            'context_hash': hash(str(sample_context.__dict__))
        }
        
        with patch.object(guidance_system, '_is_help_cache_valid', return_value=True), \
             patch.object(guidance_system, '_enhance_cached_help') as mock_enhance:
            
            mock_enhance.return_value = {"title": "Enhanced Cached Help"}
            
            result = await guidance_system.provide_contextual_help(sample_context)
            
            assert result["title"] == "Enhanced Cached Help"
            mock_enhance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_explain_error_intelligently(self, guidance_system, sample_context):
        """Test intelligent error explanation"""
        test_error = ValueError("Invalid input format")
        
        with patch.object(guidance_system, '_analyze_error') as mock_analyze, \
             patch.object(guidance_system, '_generate_error_explanation') as mock_explain, \
             patch.object(guidance_system, '_generate_actionable_solutions') as mock_solutions, \
             patch.object(guidance_system, '_assess_error_severity') as mock_severity, \
             patch.object(guidance_system, '_calculate_resolution_confidence') as mock_confidence:
            
            mock_analyze.return_value = {"error_category": "validation", "root_cause": "format"}
            mock_explain.return_value = "The input format is incorrect"
            mock_solutions.return_value = [{"title": "Fix format", "action": "validate"}]
            mock_severity.return_value = SeverityLevel.MEDIUM
            mock_confidence.return_value = 0.8
            
            result = await guidance_system.explain_error_intelligently(test_error, sample_context)
            
            assert isinstance(result, ErrorExplanation)
            assert result.error_type == "ValueError"
            assert result.user_friendly_explanation == "The input format is incorrect"
            assert result.severity == SeverityLevel.MEDIUM
            assert result.resolution_confidence == 0.8
            assert len(result.actionable_solutions) == 1
    
    @pytest.mark.asyncio
    async def test_provide_proactive_guidance(self, guidance_system):
        """Test proactive guidance provision"""
        user_id = "test_user_123"
        system_state = {"degraded_services": ["api"], "load": 0.8}
        
        with patch.object(guidance_system, '_analyze_user_behavior') as mock_behavior, \
             patch.object(guidance_system, '_identify_guidance_opportunities') as mock_opportunities, \
             patch.object(guidance_system, '_generate_proactive_guidance') as mock_generate, \
             patch.object(guidance_system, '_prioritize_guidance') as mock_prioritize, \
             patch.object(guidance_system, '_track_proactive_guidance') as mock_track:
            
            mock_behavior.return_value = UserBehaviorPattern(
                user_id=user_id,
                struggle_points=["data_upload"]
            )
            mock_opportunities.return_value = [
                {"type": "system_degradation", "priority": "high"},
                {"type": "user_struggle", "priority": "medium"}
            ]
            mock_generate.side_effect = [
                ProactiveGuidance(
                    guidance_id="guid_1",
                    user_id=user_id,
                    type=GuidanceType.PROACTIVE_SYSTEM,
                    title="System Status",
                    message="Some services are degraded",
                    priority="high"
                ),
                ProactiveGuidance(
                    guidance_id="guid_2", 
                    user_id=user_id,
                    type=GuidanceType.PROACTIVE_HELP,
                    title="Need help?",
                    message="Help with data upload",
                    priority="medium"
                )
            ]
            mock_prioritize.return_value = list(mock_generate.side_effect)
            mock_track.return_value = None
            
            result = await guidance_system.provide_proactive_guidance(user_id, system_state)
            
            assert len(result) == 2
            assert result[0].type == GuidanceType.PROACTIVE_SYSTEM
            assert result[1].type == GuidanceType.PROACTIVE_HELP
            assert result[0].priority == "high"
    
    @pytest.mark.asyncio
    async def test_create_automated_support_ticket(self, guidance_system, sample_context):
        """Test automated support ticket creation"""
        issue_description = "Unable to upload file"
        error_details = {"error_code": "UPLOAD_FAILED", "file_size": "10MB"}
        
        with patch.object(guidance_system, '_generate_ticket_context') as mock_context, \
             patch.object(guidance_system, '_assess_ticket_priority') as mock_priority, \
             patch.object(guidance_system, '_generate_ticket_title') as mock_title, \
             patch.object(guidance_system, '_generate_ticket_tags') as mock_tags, \
             patch.object(guidance_system, '_collect_relevant_attachments') as mock_attachments, \
             patch.object(guidance_system, '_attempt_automated_resolution') as mock_resolution:
            
            mock_context.return_value = {"browser": "Chrome", "os": "Windows"}
            mock_priority.return_value = "medium"
            mock_title.return_value = "File Upload Issue"
            mock_tags.return_value = ["upload", "file_handling"]
            mock_attachments.return_value = []
            mock_resolution.return_value = False
            
            result = await guidance_system.create_automated_support_ticket(
                sample_context, issue_description, error_details
            )
            
            assert isinstance(result, SupportTicket)
            assert result.title == "File Upload Issue"
            assert result.description == issue_description
            assert result.priority == "medium"
            assert result.status == TicketStatus.OPEN
            assert result.auto_created is True
            assert "upload" in result.tags
    
    @pytest.mark.asyncio
    async def test_error_explanation_with_high_severity_creates_ticket(self, guidance_system, sample_context):
        """Test that high severity errors automatically create support tickets"""
        critical_error = Exception("Database connection failed")
        
        with patch.object(guidance_system, '_analyze_error') as mock_analyze, \
             patch.object(guidance_system, '_generate_error_explanation') as mock_explain, \
             patch.object(guidance_system, '_generate_actionable_solutions') as mock_solutions, \
             patch.object(guidance_system, '_assess_error_severity') as mock_severity, \
             patch.object(guidance_system, '_calculate_resolution_confidence') as mock_confidence, \
             patch.object(guidance_system, '_should_create_support_ticket') as mock_should_create, \
             patch.object(guidance_system, '_create_automated_support_ticket') as mock_create_ticket:
            
            mock_analyze.return_value = {"error_category": "system"}
            mock_explain.return_value = "Critical system error occurred"
            mock_solutions.return_value = [{"title": "Contact support", "action": "support"}]
            mock_severity.return_value = SeverityLevel.CRITICAL
            mock_confidence.return_value = 0.2
            mock_should_create.return_value = True
            mock_create_ticket.return_value = None
            
            result = await guidance_system.explain_error_intelligently(critical_error, sample_context)
            
            assert result.severity == SeverityLevel.CRITICAL
            mock_should_create.assert_called_once()
            mock_create_ticket.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_help_on_error(self, guidance_system, sample_context):
        """Test fallback help when main system fails"""
        with patch.object(guidance_system, '_analyze_context_for_help', side_effect=Exception("Analysis failed")), \
             patch.object(guidance_system, '_provide_fallback_help') as mock_fallback:
            
            mock_fallback.return_value = {
                "type": GuidanceType.FALLBACK,
                "title": "General Help",
                "content": "Contact support for assistance"
            }
            
            result = await guidance_system.provide_contextual_help(sample_context)
            
            assert result["type"] == GuidanceType.FALLBACK
            assert result["title"] == "General Help"
            mock_fallback.assert_called_once_with(sample_context)
    
    @pytest.mark.asyncio
    async def test_user_behavior_analysis(self, guidance_system):
        """Test user behavior pattern analysis"""
        user_id = "test_user_123"
        
        with patch.object(guidance_system, '_get_common_user_actions') as mock_actions, \
             patch.object(guidance_system, '_identify_user_struggle_points') as mock_struggles, \
             patch.object(guidance_system, '_assess_user_expertise') as mock_expertise, \
             patch.object(guidance_system, '_get_preferred_help_format') as mock_format:
            
            mock_actions.return_value = ["dashboard_view", "data_upload", "report_generate"]
            mock_struggles.return_value = ["data_upload", "chart_creation"]
            mock_expertise.return_value = "intermediate"
            mock_format.return_value = "interactive"
            
            result = await guidance_system._analyze_user_behavior(user_id)
            
            assert isinstance(result, UserBehaviorPattern)
            assert result.user_id == user_id
            assert "data_upload" in result.common_actions
            assert "data_upload" in result.struggle_points
            assert result.expertise_level == "intermediate"
            assert result.preferred_help_format == "interactive"
    
    def test_help_cache_key_generation(self, guidance_system, sample_context):
        """Test help cache key generation"""
        cache_key = asyncio.run(guidance_system._generate_help_cache_key(sample_context))
        
        assert isinstance(cache_key, str)
        assert "help_" in cache_key
        assert sample_context.user_id in cache_key
        assert sample_context.current_page in cache_key
    
    def test_help_cache_validity(self, guidance_system):
        """Test help cache validity checking"""
        # Valid cache (recent)
        valid_cache = {
            'timestamp': datetime.utcnow() - timedelta(minutes=30),
            'guidance': {}
        }
        assert asyncio.run(guidance_system._is_help_cache_valid(valid_cache)) is True
        
        # Invalid cache (old)
        invalid_cache = {
            'timestamp': datetime.utcnow() - timedelta(hours=2),
            'guidance': {}
        }
        assert asyncio.run(guidance_system._is_help_cache_valid(invalid_cache)) is False
    
    @pytest.mark.asyncio
    async def test_guidance_opportunity_identification(self, guidance_system):
        """Test identification of guidance opportunities"""
        system_state = {
            "degraded_services": ["api", "database"],
            "error_rate": 0.05,
            "load": 0.9
        }
        user_patterns = UserBehaviorPattern(
            user_id="test_user",
            struggle_points=["data_export", "chart_creation"],
            expertise_level="beginner"
        )
        
        with patch.object(guidance_system, '_get_relevant_new_features') as mock_features:
            mock_features.return_value = [{"name": "Advanced Charts", "type": "visualization"}]
            
            opportunities = await guidance_system._identify_guidance_opportunities(
                system_state, user_patterns
            )
            
            assert len(opportunities) >= 3  # system degradation, user struggles, new features
            
            # Check for system degradation opportunity
            system_ops = [op for op in opportunities if op['type'] == 'system_degradation']
            assert len(system_ops) == 1
            assert system_ops[0]['priority'] == 'high'
            
            # Check for user struggle opportunities
            struggle_ops = [op for op in opportunities if op['type'] == 'user_struggle']
            assert len(struggle_ops) == 2  # Two struggle points
            
            # Check for new feature opportunities
            feature_ops = [op for op in opportunities if op['type'] == 'new_feature']
            assert len(feature_ops) == 1
    
    @pytest.mark.asyncio
    async def test_error_categorization(self, guidance_system):
        """Test error categorization functionality"""
        # Network error
        network_error = ConnectionError("Connection timeout")
        category = await guidance_system._categorize_error(network_error)
        assert category == "network"
        
        # Validation error
        validation_error = ValueError("Invalid email format")
        category = await guidance_system._categorize_error(validation_error)
        assert category == "validation"
        
        # Permission error
        permission_error = PermissionError("Access denied")
        category = await guidance_system._categorize_error(permission_error)
        assert category == "permission"
    
    @pytest.mark.asyncio
    async def test_solution_generation_by_category(self, guidance_system, sample_context):
        """Test solution generation based on error category"""
        error = ConnectionError("Network timeout")
        analysis = {"error_category": "network", "root_cause": "timeout"}
        
        solutions = await guidance_system._generate_actionable_solutions(
            error, analysis, sample_context
        )
        
        assert len(solutions) >= 1
        assert any("connection" in sol['description'].lower() for sol in solutions)
        assert any(sol['action'] == 'refresh_page' for sol in solutions)
    
    @pytest.mark.asyncio
    async def test_proactive_guidance_generation_system_degradation(self, guidance_system):
        """Test proactive guidance generation for system degradation"""
        opportunity = {
            'type': 'system_degradation',
            'priority': 'high',
            'context': ['api_service', 'database_service']
        }
        user_patterns = UserBehaviorPattern(user_id="test_user")
        system_state = {"degraded_services": ['api_service']}
        
        with patch.object(guidance_system, '_get_alternative_actions') as mock_alternatives:
            mock_alternatives.return_value = [
                {"title": "Use offline mode", "action": "enable_offline"}
            ]
            
            guidance = await guidance_system._generate_proactive_guidance(
                opportunity, user_patterns, system_state
            )
            
            assert guidance is not None
            assert guidance.type == GuidanceType.PROACTIVE_SYSTEM
            assert guidance.priority == 'high'
            assert "System Status Update" in guidance.title
            assert len(guidance.actions) >= 1

class TestUserGuidanceIntegration:
    """Integration tests for user guidance system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_help_flow(self):
        """Test complete help provision flow"""
        guidance_system = UserGuidanceSystem()
        context = GuidanceContext(
            user_id="integration_test_user",
            session_id="session_123",
            current_page="/data-upload",
            user_action="upload_file"
        )
        
        # Mock all external dependencies
        with patch.object(guidance_system, '_assess_user_experience_level', return_value="beginner"), \
             patch.object(guidance_system, '_get_common_issues_for_context', return_value=["file_size_limit"]), \
             patch.object(guidance_system, '_get_available_features', return_value=["drag_drop", "bulk_upload"]):
            
            result = await guidance_system.provide_contextual_help(context)
            
            assert result is not None
            assert 'title' in result
            assert 'content' in result
            assert 'confidence_score' in result
    
    @pytest.mark.asyncio
    async def test_error_to_ticket_flow(self):
        """Test flow from error explanation to support ticket creation"""
        guidance_system = UserGuidanceSystem()
        context = GuidanceContext(
            user_id="integration_test_user",
            session_id="session_123",
            current_page="/dashboard"
        )
        
        # Create a critical error that should trigger ticket creation
        critical_error = Exception("Critical system failure")
        
        with patch.object(guidance_system, '_assess_error_severity', return_value=SeverityLevel.CRITICAL), \
             patch.object(guidance_system, '_calculate_resolution_confidence', return_value=0.1), \
             patch.object(guidance_system, '_generate_ticket_context', return_value={}), \
             patch.object(guidance_system, '_assess_ticket_priority', return_value="critical"):
            
            explanation = await guidance_system.explain_error_intelligently(critical_error, context)
            
            assert explanation.severity == SeverityLevel.CRITICAL
            assert explanation.resolution_confidence == 0.1
            
            # Check that a support ticket was created
            assert len(guidance_system.support_tickets) == 1
            ticket = list(guidance_system.support_tickets.values())[0]
            assert ticket.priority == "critical"
            assert ticket.auto_created is True

if __name__ == "__main__":
    pytest.main([__file__])