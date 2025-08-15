"""
Tests for Transparent Status Communication System

Comprehensive tests for all components of the transparent status system:
- Status Communication Manager
- Progress Indicator Manager
- Intelligent Notification System
- Contextual Help System
- Unified Transparent Status System

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from scrollintel.core.status_communication_manager import (
    StatusCommunicationManager, SystemStatus, StatusLevel, MessageType
)
from scrollintel.core.progress_indicator_manager import (
    ProgressIndicatorManager, ProgressType, ProgressState
)
from scrollintel.core.intelligent_notification_system import (
    IntelligentNotificationSystem, NotificationChannel, NotificationPriority,
    UserActivityState, NotificationFrequency
)
from scrollintel.core.contextual_help_system import (
    ContextualHelpSystem, HelpTrigger, UserExpertiseLevel, HelpFormat
)
from scrollintel.core.transparent_status_system import (
    TransparentStatusSystem, CommunicationEvent
)


class TestStatusCommunicationManager:
    """Test Status Communication Manager"""
    
    @pytest.fixture
    def status_manager(self):
        return StatusCommunicationManager()
    
    @pytest.mark.asyncio
    async def test_update_system_status(self, status_manager):
        """Test system status updates"""
        # Test status update
        await status_manager.update_system_status(
            component="test_service",
            level=StatusLevel.DEGRADED,
            message="Service experiencing issues",
            affected_features=["feature1", "feature2"],
            alternatives=["Use alternative workflow"]
        )
        
        # Verify status was stored
        assert "test_service" in status_manager.system_status
        status = status_manager.system_status["test_service"]
        assert status.level == StatusLevel.DEGRADED
        assert status.message == "Service experiencing issues"
        assert "feature1" in status.affected_features
        assert "Use alternative workflow" in status.alternatives
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, status_manager):
        """Test progress tracking functionality"""
        operation_id = "test_operation"
        
        # Start progress tracking
        await status_manager.start_progress_tracking(
            operation_id=operation_id,
            operation_name="Test Operation",
            total_steps=5,
            can_cancel=True
        )
        
        assert operation_id in status_manager.progress_trackers
        progress = status_manager.progress_trackers[operation_id]
        assert progress.operation_name == "Test Operation"
        assert progress.total_steps == 5
        assert progress.can_cancel is True
        
        # Update progress
        await status_manager.update_progress(
            operation_id=operation_id,
            current_step=2,
            step_name="Processing data",
            partial_results={"processed": 100}
        )
        
        progress = status_manager.progress_trackers[operation_id]
        assert progress.current_step == 2
        assert progress.step_name == "Processing data"
        assert progress.progress_percentage == 40.0  # 2/5 * 100
        assert progress.partial_results["processed"] == 100
        
        # Complete progress
        await status_manager.complete_progress(
            operation_id=operation_id,
            final_results={"total_processed": 500}
        )
        
        # Should be cleaned up after delay (we won't wait for cleanup in test)
        progress = status_manager.progress_trackers[operation_id]
        assert progress.progress_percentage == 100.0
        assert progress.step_name == "Completed"
    
    @pytest.mark.asyncio
    async def test_degradation_notification(self, status_manager):
        """Test degradation notifications"""
        await status_manager.notify_degradation(
            component="api_service",
            degradation_level="partial",
            affected_functionality=["search", "filters"],
            alternatives=["Use basic search", "Browse categories"],
            explanation="API service is experiencing high load",
            estimated_recovery=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Should create notification (we'd verify this with mock in real test)
        assert len(status_manager.active_notifications) > 0
    
    @pytest.mark.asyncio
    async def test_contextual_help_provision(self, status_manager):
        """Test contextual help provision"""
        help_info = await status_manager.provide_contextual_help(
            user_id="test_user",
            user_action="upload_file",
            context={"file_type": "csv", "file_size": "large"}
        )
        
        assert help_info is not None
        assert help_info.user_action == "upload_file"
        assert help_info.confidence_score > 0
        assert len(help_info.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_system_overview(self, status_manager):
        """Test system overview generation"""
        # Add some test data
        await status_manager.update_system_status(
            component="service1",
            level=StatusLevel.OPERATIONAL,
            message="All systems operational"
        )
        
        await status_manager.start_progress_tracking(
            operation_id="op1",
            operation_name="Test Op",
            total_steps=3
        )
        
        overview = await status_manager.get_system_overview()
        
        assert "overall_status" in overview
        assert "active_operations" in overview
        assert "active_notifications" in overview
        assert "last_updated" in overview


class TestProgressIndicatorManager:
    """Test Progress Indicator Manager"""
    
    @pytest.fixture
    def progress_manager(self):
        return ProgressIndicatorManager()
    
    @pytest.mark.asyncio
    async def test_create_and_start_operation(self, progress_manager):
        """Test operation creation and starting"""
        operation_id = await progress_manager.create_operation(
            name="Data Processing",
            description="Processing uploaded data",
            progress_type=ProgressType.DETERMINATE,
            steps=[
                {"name": "Validate", "description": "Validate data format"},
                {"name": "Transform", "description": "Transform data"},
                {"name": "Store", "description": "Store processed data"}
            ],
            can_cancel=True,
            show_partial_results=True
        )
        
        assert operation_id in progress_manager.active_operations
        operation = progress_manager.active_operations[operation_id]
        assert operation.name == "Data Processing"
        assert len(operation.steps) == 3
        assert operation.can_cancel is True
        
        # Start operation
        await progress_manager.start_operation(operation_id)
        
        operation = progress_manager.active_operations[operation_id]
        assert operation.state == ProgressState.RUNNING
        assert operation.start_time is not None
        assert operation.steps[0].status == ProgressState.RUNNING
    
    @pytest.mark.asyncio
    async def test_progress_updates(self, progress_manager):
        """Test progress updates"""
        operation_id = await progress_manager.create_operation(
            name="Test Operation",
            description="Test",
            total_items=100
        )
        
        await progress_manager.start_operation(operation_id)
        
        # Update progress
        await progress_manager.update_progress(
            operation_id=operation_id,
            items_processed=25,
            message="Processing items"
        )
        
        operation = progress_manager.active_operations[operation_id]
        assert operation.items_processed == 25
        assert operation.progress_percentage == 25.0
        
        # Update with percentage
        await progress_manager.update_progress(
            operation_id=operation_id,
            progress_percentage=75.0,
            partial_results={"completed_items": 75}
        )
        
        operation = progress_manager.active_operations[operation_id]
        assert operation.progress_percentage == 75.0
        assert operation.partial_results["completed_items"] == 75
    
    @pytest.mark.asyncio
    async def test_operation_completion(self, progress_manager):
        """Test operation completion"""
        operation_id = await progress_manager.create_operation(
            name="Test Operation",
            description="Test"
        )
        
        await progress_manager.start_operation(operation_id)
        
        # Complete operation
        await progress_manager.complete_operation(
            operation_id=operation_id,
            final_results={"status": "success", "items": 100}
        )
        
        # Should be moved to completed operations
        assert operation_id not in progress_manager.active_operations
        assert operation_id in progress_manager.completed_operations
        
        operation = progress_manager.completed_operations[operation_id]
        assert operation.state == ProgressState.COMPLETED
        assert operation.progress_percentage == 100.0
        assert operation.final_results["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_operation_cancellation(self, progress_manager):
        """Test operation cancellation"""
        operation_id = await progress_manager.create_operation(
            name="Cancellable Operation",
            description="Test",
            can_cancel=True
        )
        
        await progress_manager.start_operation(operation_id)
        
        # Cancel operation
        await progress_manager.cancel_operation(
            operation_id=operation_id,
            reason="User requested cancellation"
        )
        
        # Should be moved to completed operations
        assert operation_id not in progress_manager.active_operations
        assert operation_id in progress_manager.completed_operations
        
        operation = progress_manager.completed_operations[operation_id]
        assert operation.state == ProgressState.CANCELLED
    
    @pytest.mark.asyncio
    async def test_operation_failure(self, progress_manager):
        """Test operation failure handling"""
        operation_id = await progress_manager.create_operation(
            name="Failing Operation",
            description="Test"
        )
        
        await progress_manager.start_operation(operation_id)
        
        # Fail operation
        error = Exception("Something went wrong")
        await progress_manager.fail_operation(
            operation_id=operation_id,
            error=error,
            error_details={"error_code": "E001"}
        )
        
        # Should be moved to completed operations
        assert operation_id not in progress_manager.active_operations
        assert operation_id in progress_manager.completed_operations
        
        operation = progress_manager.completed_operations[operation_id]
        assert operation.state == ProgressState.FAILED
        assert operation.error_details["error_message"] == "Something went wrong"
        assert operation.error_details["details"]["error_code"] == "E001"
    
    def test_operation_summary(self, progress_manager):
        """Test operation summary generation"""
        # This would be tested with a real operation
        # For now, test that the method exists and returns expected structure
        summary = progress_manager.get_operation_summary("nonexistent")
        assert summary is None


class TestIntelligentNotificationSystem:
    """Test Intelligent Notification System"""
    
    @pytest.fixture
    def notification_system(self):
        return IntelligentNotificationSystem()
    
    @pytest.mark.asyncio
    async def test_send_notification(self, notification_system):
        """Test sending notifications"""
        # Set up user rules first
        await notification_system.set_user_notification_rules(
            user_id="test_user",
            rules=[{
                "notification_type": "*",
                "channels": ["in_app"],
                "frequency": "immediate",
                "priority_threshold": "low"
            }]
        )
        
        result = await notification_system.send_notification(
            user_id="test_user",
            notification_type="test",
            title="Test Notification",
            message="This is a test notification",
            priority="medium"
        )
        
        assert result["status"] == "sent"
        assert "notification_id" in result
        assert "delivery_results" in result
    
    @pytest.mark.asyncio
    async def test_user_notification_rules(self, notification_system):
        """Test user notification rules"""
        rules = [{
            "notification_type": "error",
            "channels": ["in_app", "email"],
            "frequency": "immediate",
            "priority_threshold": "high",
            "quiet_hours_start": "22:00",
            "quiet_hours_end": "08:00"
        }]
        
        await notification_system.set_user_notification_rules(
            user_id="test_user",
            rules=rules
        )
        
        assert "test_user" in notification_system.user_rules
        user_rules = notification_system.user_rules["test_user"]
        assert len(user_rules) == 1
        assert user_rules[0].notification_type == "error"
        assert NotificationChannel.IN_APP in user_rules[0].channels
        assert user_rules[0].quiet_hours_start == "22:00"
    
    @pytest.mark.asyncio
    async def test_user_activity_updates(self, notification_system):
        """Test user activity state updates"""
        await notification_system.update_user_activity(
            user_id="test_user",
            activity_state=UserActivityState.ACTIVE
        )
        
        assert notification_system.user_activity["test_user"] == UserActivityState.ACTIVE
        assert "test_user" in notification_system.user_last_seen
    
    @pytest.mark.asyncio
    async def test_notification_batching(self, notification_system):
        """Test notification batching"""
        # Set up batching rules
        await notification_system.set_user_notification_rules(
            user_id="test_user",
            rules=[{
                "notification_type": "*",
                "channels": ["in_app"],
                "frequency": "batched_15min",
                "priority_threshold": "low"
            }]
        )
        
        result = await notification_system.send_notification(
            user_id="test_user",
            notification_type="info",
            title="Batched Notification",
            message="This should be batched",
            priority="low"
        )
        
        assert result["status"] == "batched"
        assert "batch_id" in result
        assert "scheduled_delivery" in result
        
        # Check that batch was created
        assert "test_user" in notification_system.pending_batches
        assert len(notification_system.pending_batches["test_user"]) > 0
    
    @pytest.mark.asyncio
    async def test_notification_stats(self, notification_system):
        """Test notification statistics"""
        stats = await notification_system.get_user_notification_stats("test_user")
        
        assert "user_id" in stats
        assert "total_sent" in stats
        assert "delivery_rate" in stats
        assert "recent_deliveries" in stats
    
    @pytest.mark.asyncio
    async def test_notification_analytics(self, notification_system):
        """Test system-wide notification analytics"""
        analytics = await notification_system.get_notification_analytics()
        
        assert "total_users" in analytics
        assert "total_notifications" in analytics
        assert "channel_performance" in analytics
        assert "last_updated" in analytics


class TestContextualHelpSystem:
    """Test Contextual Help System"""
    
    @pytest.fixture
    def help_system(self):
        return ContextualHelpSystem()
    
    @pytest.mark.asyncio
    async def test_get_contextual_help(self, help_system):
        """Test getting contextual help"""
        suggestions = await help_system.get_contextual_help(
            user_id="test_user",
            trigger=HelpTrigger.USER_REQUEST,
            context={"page": "upload", "action": "file_upload"},
            max_suggestions=3
        )
        
        assert isinstance(suggestions, list)
        # Should have suggestions based on default content
        assert len(suggestions) > 0
        
        for suggestion in suggestions:
            assert hasattr(suggestion, 'suggestion_id')
            assert hasattr(suggestion, 'content')
            assert hasattr(suggestion, 'relevance_score')
            assert suggestion.relevance_score >= 0
    
    @pytest.mark.asyncio
    async def test_proactive_help(self, help_system):
        """Test proactive help provision"""
        # First, update user context to simulate confusion
        await help_system._update_user_context(
            user_id="test_user",
            context={"page": "visualization", "action": "create_chart"}
        )
        
        # Simulate repeated actions (confusion signal)
        user_context = help_system.user_contexts["test_user"]
        user_context.recent_actions = ["create_chart", "create_chart", "create_chart"]
        
        suggestion = await help_system.provide_proactive_help(
            user_id="test_user",
            current_context={"page": "visualization"}
        )
        
        # May or may not have suggestion depending on confusion detection
        if suggestion:
            assert hasattr(suggestion, 'suggestion_id')
            assert hasattr(suggestion, 'content')
    
    @pytest.mark.asyncio
    async def test_error_help(self, help_system):
        """Test error-specific help"""
        error_info = {
            "type": "ValidationError",
            "message": "File format not supported"
        }
        
        suggestions = await help_system.handle_error_help(
            user_id="test_user",
            error=error_info,
            context={"page": "upload", "file_type": "unknown"}
        )
        
        assert isinstance(suggestions, list)
        # Should provide relevant help for file upload errors
    
    @pytest.mark.asyncio
    async def test_add_help_content(self, help_system):
        """Test adding help content"""
        await help_system.add_help_content(
            content_id="test_content",
            title="Test Help Content",
            content="This is test help content for testing purposes.",
            format=HelpFormat.TEXT,
            expertise_level=UserExpertiseLevel.BEGINNER,
            tags=["test", "help"]
        )
        
        assert "test_content" in help_system.help_content
        content = help_system.help_content["test_content"]
        assert content.title == "Test Help Content"
        assert content.format == HelpFormat.TEXT
        assert "test" in content.tags
    
    @pytest.mark.asyncio
    async def test_interactive_tutorial(self, help_system):
        """Test interactive tutorial functionality"""
        # Add tutorial content
        await help_system.add_help_content(
            content_id="test_tutorial",
            title="Test Tutorial",
            content="Interactive tutorial for testing",
            format=HelpFormat.INTERACTIVE,
            expertise_level=UserExpertiseLevel.BEGINNER,
            interactive_elements=[
                {"type": "highlight", "target": "#button", "instructions": "Click this button"},
                {"type": "form", "target": "#input", "instructions": "Enter your data"}
            ]
        )
        
        # Start tutorial
        session = await help_system.start_interactive_tutorial(
            user_id="test_user",
            tutorial_id="test_tutorial"
        )
        
        assert "session_id" in session
        assert "tutorial" in session
        assert "current_step" in session
        assert session["tutorial"]["total_steps"] == 2
        
        # Advance tutorial
        result = await help_system.advance_tutorial(
            session_id=session["session_id"],
            step_result={"success": True}
        )
        
        assert "current_step" in result
        assert result["progress"] > 0
    
    @pytest.mark.asyncio
    async def test_help_feedback(self, help_system):
        """Test help feedback recording"""
        await help_system.record_help_feedback(
            user_id="test_user",
            content_id="getting_started",  # Default content
            helpful=True,
            feedback="Very helpful tutorial"
        )
        
        # Check that effectiveness was updated
        assert "getting_started" in help_system.content_effectiveness
        effectiveness = help_system.content_effectiveness["getting_started"]
        assert effectiveness["total"] == 1
        assert effectiveness["helpful"] == 1
    
    @pytest.mark.asyncio
    async def test_help_analytics(self, help_system):
        """Test help analytics"""
        analytics = await help_system.get_help_analytics()
        
        assert "total_content" in analytics
        assert "total_users" in analytics
        assert "content_effectiveness" in analytics
        assert "average_effectiveness" in analytics


class TestTransparentStatusSystem:
    """Test Unified Transparent Status System"""
    
    @pytest.fixture
    def status_system(self):
        return TransparentStatusSystem()
    
    @pytest.mark.asyncio
    async def test_communicate_status_change(self, status_system):
        """Test unified status change communication"""
        communication_id = await status_system.communicate_status_change(
            component="test_service",
            status_level=StatusLevel.DEGRADED,
            message="Service experiencing issues",
            affected_features=["search", "filters"],
            alternatives=["Use basic search"]
        )
        
        assert communication_id is not None
        assert communication_id in status_system.active_communications
        
        communication = status_system.active_communications[communication_id]
        assert communication.event_type == CommunicationEvent.STATUS_CHANGE
        assert communication.component == "test_service"
        assert communication.priority == "medium"  # Degraded maps to medium
    
    @pytest.mark.asyncio
    async def test_operation_tracking_integration(self, status_system):
        """Test integrated operation tracking"""
        # Start operation
        operation_id = await status_system.start_operation_tracking(
            user_id="test_user",
            operation_name="Data Processing",
            description="Processing uploaded data",
            total_steps=3,
            can_cancel=True
        )
        
        assert operation_id is not None
        
        # Update progress
        await status_system.update_operation_progress(
            operation_id=operation_id,
            user_id="test_user",
            progress_percentage=50.0,
            step_name="Processing data"
        )
        
        # Complete operation
        await status_system.complete_operation(
            operation_id=operation_id,
            user_id="test_user",
            success=True,
            final_results={"processed_items": 100}
        )
        
        # Verify operation was completed
        operation = status_system.progress_manager.get_operation(operation_id)
        assert operation is not None
        assert operation.state == ProgressState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_error_handling_with_help(self, status_system):
        """Test integrated error handling with help"""
        error = Exception("File upload failed")
        context = {"page": "upload", "file_type": "csv"}
        
        help_suggestions = await status_system.handle_error_with_help(
            user_id="test_user",
            error=error,
            context=context,
            component="file_upload"
        )
        
        assert isinstance(help_suggestions, list)
        # Should provide relevant help suggestions
    
    @pytest.mark.asyncio
    async def test_proactive_help_integration(self, status_system):
        """Test integrated proactive help"""
        context = {"page": "dashboard", "idle_time": 600}  # 10 minutes idle
        
        suggestion = await status_system.provide_proactive_help(
            user_id="test_user",
            context=context
        )
        
        # May or may not have suggestion
        if suggestion:
            assert "id" in suggestion
            assert "title" in suggestion
            assert "content" in suggestion
    
    @pytest.mark.asyncio
    async def test_user_activity_integration(self, status_system):
        """Test user activity state integration"""
        await status_system.update_user_activity(
            user_id="test_user",
            activity_state=UserActivityState.ACTIVE,
            context={"page": "dashboard"}
        )
        
        # Should update across all systems
        assert status_system.notification_system.user_activity["test_user"] == UserActivityState.ACTIVE
    
    @pytest.mark.asyncio
    async def test_user_preferences_integration(self, status_system):
        """Test user preferences integration"""
        preferences = {
            "notification_rules": [{
                "notification_type": "*",
                "channels": ["in_app"],
                "frequency": "immediate",
                "priority_threshold": "medium"
            }],
            "expertise_level": "intermediate"
        }
        
        await status_system.set_user_communication_preferences(
            user_id="test_user",
            preferences=preferences
        )
        
        # Should update across systems
        assert "test_user" in status_system.notification_system.user_rules
        assert "test_user" in status_system.help_system.user_contexts
        
        user_context = status_system.help_system.user_contexts["test_user"]
        assert user_context.expertise_level == UserExpertiseLevel.INTERMEDIATE
    
    @pytest.mark.asyncio
    async def test_user_status_overview(self, status_system):
        """Test comprehensive user status overview"""
        overview = await status_system.get_user_status_overview("test_user")
        
        assert "user_id" in overview
        assert "system_status" in overview
        assert "active_operations" in overview
        assert "notification_stats" in overview
        assert "help_available" in overview
        assert "last_updated" in overview
    
    @pytest.mark.asyncio
    async def test_communication_subscription(self, status_system):
        """Test communication subscription"""
        received_communications = []
        
        async def communication_handler(status_update):
            received_communications.append(status_update)
        
        # Subscribe to user communications
        await status_system.subscribe_to_user_communications(
            user_id="test_user",
            callback=communication_handler
        )
        
        # Send a communication
        await status_system.communicate_status_change(
            component="test_service",
            status_level=StatusLevel.OPERATIONAL,
            message="Service restored"
        )
        
        # Should have received communication (in real implementation)
        # For now, just verify subscription was set up
        assert "test_user" in status_system.user_subscribers
        assert len(status_system.user_subscribers["test_user"]) == 1


# Integration Tests

class TestSystemIntegration:
    """Test full system integration scenarios"""
    
    @pytest.fixture
    def status_system(self):
        return TransparentStatusSystem()
    
    @pytest.mark.asyncio
    async def test_complete_user_workflow(self, status_system):
        """Test complete user workflow with all components"""
        user_id = "integration_test_user"
        
        # 1. Set user preferences
        await status_system.set_user_communication_preferences(
            user_id=user_id,
            preferences={
                "notification_rules": [{
                    "notification_type": "*",
                    "channels": ["in_app"],
                    "frequency": "immediate",
                    "priority_threshold": "low"
                }],
                "expertise_level": "beginner"
            }
        )
        
        # 2. Update user activity
        await status_system.update_user_activity(
            user_id=user_id,
            activity_state=UserActivityState.ACTIVE,
            context={"page": "upload"}
        )
        
        # 3. Start an operation
        operation_id = await status_system.start_operation_tracking(
            user_id=user_id,
            operation_name="File Upload",
            description="Uploading user data",
            total_steps=3
        )
        
        # 4. Update progress
        await status_system.update_operation_progress(
            operation_id=operation_id,
            user_id=user_id,
            progress_percentage=33.0,
            step_name="Validating file"
        )
        
        # 5. Simulate an error
        error = Exception("Invalid file format")
        help_suggestions = await status_system.handle_error_with_help(
            user_id=user_id,
            error=error,
            context={"page": "upload", "file_type": "unknown"}
        )
        
        # 6. Get proactive help
        proactive_help = await status_system.provide_proactive_help(
            user_id=user_id,
            context={"page": "upload", "error_occurred": True}
        )
        
        # 7. Get user overview
        overview = await status_system.get_user_status_overview(user_id)
        
        # Verify the workflow worked
        assert overview["user_id"] == user_id
        assert len(overview["active_operations"]) > 0
        assert overview["notification_stats"]["user_id"] == user_id
        
        # Clean up - complete the operation
        await status_system.complete_operation(
            operation_id=operation_id,
            user_id=user_id,
            success=False,
            message="Operation failed due to invalid file format"
        )
    
    @pytest.mark.asyncio
    async def test_system_degradation_scenario(self, status_system):
        """Test system degradation scenario with user communication"""
        # Simulate system degradation
        communication_id = await status_system.communicate_status_change(
            component="database",
            status_level=StatusLevel.DEGRADED,
            message="Database experiencing high load",
            affected_features=["search", "advanced_filters"],
            alternatives=["Use basic search", "Browse by category"],
            estimated_resolution=datetime.utcnow() + timedelta(hours=2)
        )
        
        # Verify communication was created
        assert communication_id in status_system.active_communications
        
        # Get system overview
        overview = await status_system.status_manager.get_system_overview()
        assert overview["overall_status"] == "degraded"
        
        # Simulate recovery
        await status_system.communicate_status_change(
            component="database",
            status_level=StatusLevel.OPERATIONAL,
            message="Database performance restored"
        )
        
        # Verify recovery
        overview = await status_system.status_manager.get_system_overview()
        assert overview["overall_status"] == "operational"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])