"""
Demo: Transparent Status Communication System

This demo showcases the comprehensive transparent status communication system
including status updates, progress tracking, intelligent notifications,
and contextual help.

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import asyncio
import logging
from datetime import datetime, timedelta
import json
import time

from scrollintel.core.transparent_status_system import (
    transparent_status_system, CommunicationEvent
)
from scrollintel.core.status_communication_manager import StatusLevel
from scrollintel.core.intelligent_notification_system import (
    UserActivityState, NotificationChannel
)
from scrollintel.core.contextual_help_system import (
    HelpTrigger, UserExpertiseLevel, HelpFormat
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransparentStatusDemo:
    """Demo class for transparent status communication system"""
    
    def __init__(self):
        self.system = transparent_status_system
        self.demo_users = ["alice", "bob", "charlie"]
        self.received_communications = []
    
    async def setup_demo(self):
        """Set up demo environment"""
        print("🚀 Setting up Transparent Status Communication System Demo")
        print("=" * 60)
        
        # Set up communication handler
        await self.system.subscribe_to_all_communications(self.communication_handler)
        
        # Set up user preferences
        for user_id in self.demo_users:
            await self.setup_user_preferences(user_id)
        
        # Add custom help content
        await self.add_demo_help_content()
        
        print("✅ Demo setup complete!\n")
    
    async def setup_user_preferences(self, user_id: str):
        """Set up preferences for demo user"""
        expertise_map = {
            "alice": "beginner",
            "bob": "intermediate", 
            "charlie": "expert"
        }
        
        preferences = {
            "notification_rules": [{
                "notification_type": "*",
                "channels": ["in_app"],
                "frequency": "immediate",
                "priority_threshold": "low"
            }],
            "expertise_level": expertise_map.get(user_id, "beginner")
        }
        
        await self.system.set_user_communication_preferences(user_id, preferences)
        print(f"👤 Set up preferences for {user_id} ({expertise_map.get(user_id, 'beginner')} level)")
    
    async def add_demo_help_content(self):
        """Add demo help content"""
        help_contents = [
            {
                "content_id": "file_upload_demo",
                "title": "File Upload Guide",
                "content": "Learn how to upload files efficiently. Supported formats: CSV, Excel, JSON. Max size: 100MB.",
                "format": HelpFormat.TEXT,
                "expertise_level": UserExpertiseLevel.BEGINNER,
                "tags": ["upload", "files", "getting-started"]
            },
            {
                "content_id": "data_analysis_demo",
                "title": "Advanced Data Analysis",
                "content": "Master advanced data analysis techniques including statistical modeling and machine learning.",
                "format": HelpFormat.MARKDOWN,
                "expertise_level": UserExpertiseLevel.ADVANCED,
                "tags": ["analysis", "statistics", "machine-learning"]
            },
            {
                "content_id": "error_recovery_demo",
                "title": "Error Recovery Guide",
                "content": "How to recover from common errors and troubleshoot issues effectively.",
                "format": HelpFormat.STEP_BY_STEP,
                "expertise_level": UserExpertiseLevel.INTERMEDIATE,
                "tags": ["troubleshooting", "errors", "recovery"]
            }
        ]
        
        for content_data in help_contents:
            await self.system.help_system.add_help_content(**content_data)
        
        print(f"📚 Added {len(help_contents)} help content items")
    
    async def communication_handler(self, status_update):
        """Handle received communications"""
        self.received_communications.append(status_update)
        
        print(f"📢 Communication: {status_update.title}")
        print(f"   Type: {status_update.event_type.value}")
        print(f"   Component: {status_update.component}")
        print(f"   Priority: {status_update.priority}")
        print(f"   Message: {status_update.message}")
        if status_update.user_id:
            print(f"   User: {status_update.user_id}")
        print()
    
    async def demo_system_status_updates(self):
        """Demo system status updates"""
        print("📊 Demo: System Status Updates")
        print("-" * 40)
        
        # Normal operation
        await self.system.communicate_status_change(
            component="api_service",
            status_level=StatusLevel.OPERATIONAL,
            message="All API services running normally"
        )
        
        await asyncio.sleep(1)
        
        # Service degradation
        await self.system.communicate_status_change(
            component="database",
            status_level=StatusLevel.DEGRADED,
            message="Database experiencing high load - some queries may be slower",
            affected_features=["advanced_search", "real_time_updates"],
            alternatives=["Use basic search", "Refresh manually for updates"],
            estimated_resolution=datetime.utcnow() + timedelta(minutes=30)
        )
        
        await asyncio.sleep(1)
        
        # Maintenance mode
        await self.system.communicate_status_change(
            component="file_storage",
            status_level=StatusLevel.MAINTENANCE,
            message="Scheduled maintenance in progress - file uploads temporarily unavailable",
            affected_features=["file_upload", "file_download"],
            alternatives=["Save work locally", "Try again in 15 minutes"],
            estimated_resolution=datetime.utcnow() + timedelta(minutes=15)
        )
        
        await asyncio.sleep(1)
        
        # Recovery
        await self.system.communicate_status_change(
            component="database",
            status_level=StatusLevel.RECOVERING,
            message="Database performance improving - full service restoration in progress"
        )
        
        await asyncio.sleep(1)
        
        # Full recovery
        await self.system.communicate_status_change(
            component="database",
            status_level=StatusLevel.OPERATIONAL,
            message="Database performance fully restored"
        )
        
        print("✅ System status updates demo complete\n")
    
    async def demo_progress_tracking(self):
        """Demo progress tracking"""
        print("⏳ Demo: Progress Tracking")
        print("-" * 40)
        
        user_id = "alice"
        
        # Update user activity
        await self.system.update_user_activity(
            user_id=user_id,
            activity_state=UserActivityState.ACTIVE,
            context={"page": "data_processing"}
        )
        
        # Start operation
        operation_id = await self.system.start_operation_tracking(
            user_id=user_id,
            operation_name="Data Processing Pipeline",
            description="Processing uploaded dataset with advanced analytics",
            total_steps=5,
            estimated_duration=120,  # 2 minutes
            can_cancel=True,
            show_progress=True
        )
        
        print(f"🔄 Started operation: {operation_id}")
        
        # Simulate progress updates
        steps = [
            (20, "Validating data format"),
            (40, "Cleaning and preprocessing"),
            (60, "Running statistical analysis"),
            (80, "Generating visualizations"),
            (100, "Finalizing results")
        ]
        
        for progress, step_name in steps:
            await asyncio.sleep(0.5)  # Simulate processing time
            
            await self.system.update_operation_progress(
                operation_id=operation_id,
                user_id=user_id,
                progress_percentage=progress,
                step_name=step_name,
                partial_results={"processed_rows": progress * 10}
            )
            
            print(f"   Progress: {progress}% - {step_name}")
        
        # Complete operation
        await self.system.complete_operation(
            operation_id=operation_id,
            user_id=user_id,
            success=True,
            final_results={
                "total_rows_processed": 1000,
                "insights_generated": 15,
                "visualizations_created": 8
            },
            message="Data processing completed successfully"
        )
        
        print("✅ Progress tracking demo complete\n")
    
    async def demo_error_handling_with_help(self):
        """Demo error handling with contextual help"""
        print("🚨 Demo: Error Handling with Contextual Help")
        print("-" * 40)
        
        user_id = "bob"
        
        # Update user activity
        await self.system.update_user_activity(
            user_id=user_id,
            activity_state=UserActivityState.ACTIVE,
            context={"page": "file_upload"}
        )
        
        # Simulate file upload error
        error = Exception("File format not supported. Please use CSV, Excel, or JSON format.")
        context = {
            "page": "file_upload",
            "file_name": "data.xyz",
            "file_size": "5MB",
            "attempted_action": "upload_file"
        }
        
        help_suggestions = await self.system.handle_error_with_help(
            user_id=user_id,
            error=error,
            context=context,
            component="file_upload_service"
        )
        
        print(f"💡 Provided {len(help_suggestions)} help suggestions:")
        for i, suggestion in enumerate(help_suggestions, 1):
            print(f"   {i}. {suggestion['title']} (relevance: {suggestion['relevance']:.2f})")
        
        print("✅ Error handling with help demo complete\n")
    
    async def demo_proactive_help(self):
        """Demo proactive help system"""
        print("🤖 Demo: Proactive Help System")
        print("-" * 40)
        
        user_id = "charlie"
        
        # Simulate user getting stuck (repeated actions)
        await self.system.update_user_activity(
            user_id=user_id,
            activity_state=UserActivityState.ACTIVE,
            context={"page": "visualization"}
        )
        
        # Simulate confusion signals
        help_system = self.system.help_system
        if user_id not in help_system.user_contexts:
            await help_system._update_user_context(user_id, {"page": "visualization"})
        
        user_context = help_system.user_contexts[user_id]
        user_context.recent_actions = ["create_chart", "create_chart", "create_chart", "delete_chart", "create_chart"]
        
        # Get proactive help
        suggestion = await self.system.provide_proactive_help(
            user_id=user_id,
            context={
                "page": "visualization",
                "idle_time": 300,  # 5 minutes idle
                "repeated_actions": True
            }
        )
        
        if suggestion:
            print(f"💡 Proactive help suggestion: {suggestion['title']}")
            print(f"   Relevance: {suggestion['relevance']:.2f}")
            print(f"   Content preview: {suggestion['content'][:100]}...")
        else:
            print("ℹ️  No proactive help needed at this time")
        
        print("✅ Proactive help demo complete\n")
    
    async def demo_interactive_tutorial(self):
        """Demo interactive tutorial system"""
        print("🎓 Demo: Interactive Tutorial System")
        print("-" * 40)
        
        user_id = "alice"
        
        # Add interactive tutorial
        await self.system.help_system.add_help_content(
            content_id="data_upload_tutorial",
            title="Data Upload Tutorial",
            content="Learn how to upload and process your data step by step",
            format=HelpFormat.INTERACTIVE,
            expertise_level=UserExpertiseLevel.BEGINNER,
            tags=["tutorial", "upload", "interactive"],
            interactive_elements=[
                {
                    "type": "highlight",
                    "target": "#upload-button",
                    "instructions": "Click the upload button to start"
                },
                {
                    "type": "form",
                    "target": "#file-input",
                    "instructions": "Select your data file"
                },
                {
                    "type": "wait",
                    "target": "#progress-bar",
                    "instructions": "Wait for upload to complete"
                }
            ]
        )
        
        # Start tutorial
        session = await self.system.help_system.start_interactive_tutorial(
            user_id=user_id,
            tutorial_id="data_upload_tutorial",
            context={"page": "upload"}
        )
        
        print(f"📖 Started tutorial: {session['tutorial']['title']}")
        print(f"   Total steps: {session['tutorial']['total_steps']}")
        print(f"   Current step: {session['current_step']['instructions']}")
        
        # Simulate tutorial progression
        for step in range(session['tutorial']['total_steps']):
            await asyncio.sleep(0.5)
            
            result = await self.system.help_system.advance_tutorial(
                session_id=session['session_id'],
                step_result={"success": True, "time_taken": 5}
            )
            
            if result['completed']:
                print(f"🎉 Tutorial completed! Progress: {result['progress']}%")
                break
            else:
                print(f"   Step {step + 2}: {result['current_step']['instructions']} (Progress: {result['progress']:.1f}%)")
        
        print("✅ Interactive tutorial demo complete\n")
    
    async def demo_notification_intelligence(self):
        """Demo intelligent notification system"""
        print("🔔 Demo: Intelligent Notification System")
        print("-" * 40)
        
        # Set up different notification preferences for users
        notification_scenarios = [
            {
                "user_id": "alice",
                "rules": [{
                    "notification_type": "*",
                    "channels": ["in_app"],
                    "frequency": "immediate",
                    "priority_threshold": "low"
                }]
            },
            {
                "user_id": "bob", 
                "rules": [{
                    "notification_type": "*",
                    "channels": ["in_app"],
                    "frequency": "batched_15min",
                    "priority_threshold": "medium"
                }]
            },
            {
                "user_id": "charlie",
                "rules": [{
                    "notification_type": "*",
                    "channels": ["in_app"],
                    "frequency": "immediate",
                    "priority_threshold": "high",
                    "quiet_hours_start": "22:00",
                    "quiet_hours_end": "08:00"
                }]
            }
        ]
        
        # Set up notification rules
        for scenario in notification_scenarios:
            await self.system.notification_system.set_user_notification_rules(
                user_id=scenario["user_id"],
                rules=scenario["rules"]
            )
            print(f"📋 Set notification rules for {scenario['user_id']}")
        
        # Send different types of notifications
        notifications = [
            {
                "user_id": "alice",
                "type": "info",
                "title": "New Feature Available",
                "message": "Check out our new data visualization tools!",
                "priority": "low"
            },
            {
                "user_id": "bob",
                "type": "warning", 
                "title": "Storage Almost Full",
                "message": "Your storage is 85% full. Consider cleaning up old files.",
                "priority": "medium"
            },
            {
                "user_id": "charlie",
                "type": "error",
                "title": "Processing Failed",
                "message": "Your data processing job failed. Click for details.",
                "priority": "high"
            }
        ]
        
        for notif in notifications:
            result = await self.system.notification_system.send_notification(
                user_id=notif["user_id"],
                notification_type=notif["type"],
                title=notif["title"],
                message=notif["message"],
                priority=notif["priority"]
            )
            
            print(f"📨 Sent {notif['type']} notification to {notif['user_id']}: {result['status']}")
        
        print("✅ Intelligent notification demo complete\n")
    
    async def demo_user_status_overview(self):
        """Demo user status overview"""
        print("📈 Demo: User Status Overview")
        print("-" * 40)
        
        for user_id in self.demo_users:
            overview = await self.system.get_user_status_overview(user_id)
            
            print(f"👤 Status overview for {user_id}:")
            print(f"   System status: {overview['system_status']['overall_status']}")
            print(f"   Active operations: {len(overview['active_operations'])}")
            print(f"   Notifications sent: {overview['notification_stats']['total_sent']}")
            print(f"   Help content available: {overview['help_available']}")
            print()
        
        print("✅ User status overview demo complete\n")
    
    async def demo_analytics(self):
        """Demo system analytics"""
        print("📊 Demo: System Analytics")
        print("-" * 40)
        
        # Get notification analytics
        notif_analytics = await self.system.notification_system.get_notification_analytics()
        print("📧 Notification Analytics:")
        print(f"   Total users: {notif_analytics['total_users']}")
        print(f"   Total notifications: {notif_analytics['total_notifications']}")
        print(f"   Active batches: {notif_analytics['active_batches']}")
        
        # Get help analytics
        help_analytics = await self.system.help_system.get_help_analytics()
        print("\n📚 Help System Analytics:")
        print(f"   Total content: {help_analytics['total_content']}")
        print(f"   Total users: {help_analytics['total_users']}")
        print(f"   Average effectiveness: {help_analytics['average_effectiveness']:.2f}")
        
        print("\n✅ Analytics demo complete\n")
    
    async def run_complete_demo(self):
        """Run the complete demo"""
        try:
            await self.setup_demo()
            
            # Run all demo scenarios
            await self.demo_system_status_updates()
            await self.demo_progress_tracking()
            await self.demo_error_handling_with_help()
            await self.demo_proactive_help()
            await self.demo_interactive_tutorial()
            await self.demo_notification_intelligence()
            await self.demo_user_status_overview()
            await self.demo_analytics()
            
            # Summary
            print("📋 Demo Summary")
            print("=" * 60)
            print(f"✅ Total communications received: {len(self.received_communications)}")
            print(f"✅ Demo users configured: {len(self.demo_users)}")
            print(f"✅ Help content items added: {len(self.system.help_system.help_content)}")
            print(f"✅ All transparent status system features demonstrated!")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise


async def main():
    """Main demo function"""
    print("🌟 Transparent Status Communication System Demo")
    print("=" * 60)
    print("This demo showcases:")
    print("• Real-time status communication")
    print("• Intelligent progress tracking")
    print("• Smart notification system")
    print("• Contextual help and guidance")
    print("• Integrated error handling")
    print("• Proactive user assistance")
    print("• Interactive tutorials")
    print("• System analytics")
    print()
    
    demo = TransparentStatusDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())