"""
Demo: Message Coordination Engine

Demonstrates the crisis message coordination system including consistent messaging,
approval workflows, version control, and effectiveness tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.engines.message_coordination_engine import (
    MessageCoordinationEngine, MessageStatus, ApprovalStatus
)
from scrollintel.models.crisis_communication_models import (
    NotificationChannel, NotificationPriority, StakeholderType
)


async def demonstrate_message_coordination():
    """Demonstrate comprehensive message coordination capabilities"""
    print("\n" + "="*80)
    print("CRISIS MESSAGE COORDINATION SYSTEM DEMO")
    print("="*80)
    
    # Initialize the coordination engine
    print("\n1. Initializing Message Coordination Engine...")
    engine = MessageCoordinationEngine()
    
    print(f"   ✓ Channel adapters loaded: {len(engine.channel_adapters)}")
    print(f"   ✓ Supported channels: {', '.join(engine.channel_adapters.keys())}")
    
    # Demonstrate creating coordinated message
    print("\n2. Creating Coordinated Crisis Message...")
    
    crisis_message_data = {
        "crisis_id": "crisis_security_001",
        "message_type": "initial_alert",
        "master_content": """URGENT SECURITY ALERT

We have detected a security incident affecting our user authentication system. 

IMMEDIATE ACTIONS TAKEN:
• Affected systems have been isolated
• Security team has been activated
• Investigation is underway

IMPACT:
• User login may be temporarily unavailable
• No customer data has been compromised
• All other services remain operational

NEXT STEPS:
• We will provide updates every 30 minutes
• Full service restoration expected within 2 hours
• Post-incident report will be published within 24 hours

For immediate assistance, contact our support team at support@company.com

We apologize for any inconvenience and appreciate your patience.""",
        "master_subject": "🚨 URGENT: Security Incident - Immediate Action Required",
        "target_channels": [
            NotificationChannel.EMAIL,
            NotificationChannel.SMS,
            NotificationChannel.SLACK,
            NotificationChannel.PORTAL
        ],
        "target_stakeholders": [
            StakeholderType.EXECUTIVE,
            StakeholderType.EMPLOYEE,
            StakeholderType.CUSTOMER
        ],
        "priority": NotificationPriority.CRITICAL,
        "requires_approval": True,
        "created_by": "crisis_manager"
    }
    
    message = await engine.create_coordinated_message(**crisis_message_data)
    
    print(f"   ✓ Message created: {message.id}")
    print(f"   📧 Target channels: {len(message.target_channels)}")
    print(f"   👥 Target stakeholders: {len(message.target_stakeholders)}")
    print(f"   📋 Status: {message.status.value}")
    print(f"   🔄 Requires approval: {message.requires_approval}")
    print(f"   📝 Current version: {message.current_version}")
    
    # Show channel adaptations
    print("\n3. Demonstrating Channel-Specific Adaptations...")
    
    for channel_key, adaptation in message.channel_adaptations.items():
        print(f"\n   📱 {channel_key.upper()} Adaptation:")
        print(f"      Subject: {adaptation['subject'][:50]}{'...' if len(adaptation['subject']) > 50 else ''}")
        print(f"      Content length: {len(adaptation['content'])} chars")
        print(f"      Truncated: {adaptation['metadata']['truncated']}")
        
        if channel_key == "sms":
            print(f"      SMS Preview: {adaptation['content'][:100]}...")
        elif channel_key == "slack":
            print(f"      Slack Preview: {adaptation['content'][:100]}...")
    
    # Demonstrate approval workflow
    print("\n4. Approval Workflow Management...")
    
    print(f"   📋 Approval steps required: {len(message.approval_workflow)}")
    for i, step in enumerate(message.approval_workflow, 1):
        print(f"   {i}. {step.approver_role} - Status: {step.status.value}")
        print(f"      Timeout: {step.escalation_timeout_minutes} minutes")
    
    # Submit for approval
    print("\n   Submitting message for approval...")
    success = await engine.submit_for_approval(message.id)
    print(f"   ✓ Submission successful: {success}")
    
    # Simulate approval process
    print("\n5. Simulating Approval Process...")
    
    for step in message.approval_workflow:
        if step.status == ApprovalStatus.PENDING:
            print(f"\n   Approving step: {step.approver_role}")
            
            success = await engine.approve_message(
                message.id,
                f"approver_{step.step_order}",
                step.approver_role,
                f"Approved by {step.approver_role} - Content looks good"
            )
            
            print(f"   ✓ Approval result: {success}")
            print(f"   📝 Comments: {step.comments}")
            
            # Check if message is fully approved
            if message.status == MessageStatus.APPROVED:
                print(f"   🎉 Message fully approved!")
                break
    
    # Demonstrate version control
    print("\n6. Version Control Demonstration...")
    
    print(f"   Current versions: {len(message.versions)}")
    for version in message.versions:
        print(f"   📄 Version {version.version_number} by {version.author}")
        print(f"      Created: {version.created_at.strftime('%H:%M:%S')}")
        print(f"      Current: {version.is_current}")
        print(f"      Changes: {version.changes_summary}")
    
    # Create new version
    print("\n   Creating updated version...")
    updated_content = message.master_content + "\n\nUPDATE: Additional security measures have been implemented."
    
    success = await engine.update_message_version(
        message.id,
        updated_content,
        message.master_subject + " - Updated",
        "security_team",
        "Added security update information"
    )
    
    print(f"   ✓ Version update successful: {success}")
    print(f"   📝 New version: {message.current_version}")
    print(f"   📋 Status reset to: {message.status.value}")
    
    # Re-approve the updated message
    print("\n   Re-approving updated message...")
    for step in message.approval_workflow:
        if step.status == ApprovalStatus.PENDING:
            await engine.approve_message(
                message.id,
                f"approver_{step.step_order}",
                step.approver_role,
                "Updated version approved"
            )
    
    # Publish message
    print("\n7. Publishing Message to All Channels...")
    
    if message.status == MessageStatus.APPROVED:
        result = await engine.publish_message(message.id)
        
        if result["success"]:
            print(f"   ✓ Message published successfully!")
            print(f"   📅 Published at: {result['published_at']}")
            print(f"   📊 Publication results:")
            
            for channel, channel_result in result["channels"].items():
                status_icon = "✅" if channel_result["success"] else "❌"
                print(f"      {status_icon} {channel}: {channel_result.get('sent_at', 'Failed')}")
        else:
            print(f"   ❌ Publication failed: {result['error']}")
    
    # Track effectiveness
    print("\n8. Message Effectiveness Tracking...")
    
    metrics = await engine.track_message_effectiveness(message.id)
    
    print(f"   📊 Effectiveness Metrics:")
    print(f"      Total sent: {metrics.total_sent}")
    print(f"      Delivered: {metrics.total_delivered} ({metrics.delivery_rate:.1%})")
    print(f"      Read: {metrics.total_read} ({metrics.read_rate:.1%})")
    print(f"      Responded: {metrics.total_responded} ({metrics.response_rate:.1%})")
    print(f"      Overall effectiveness: {metrics.overall_effectiveness_score:.2f}")
    
    print(f"\n   📱 Channel Performance:")
    for channel, performance in metrics.channel_performance.items():
        print(f"      {channel}: Delivery {performance['delivery_rate']:.1%}, "
              f"Read {performance['read_rate']:.1%}, "
              f"Response {performance['response_rate']:.1%}")
    
    # Demonstrate content adaptation testing
    print("\n9. Content Adaptation Testing...")
    
    test_content = """🚨 CRITICAL SYSTEM ALERT 🚨

Our primary database cluster is experiencing high latency issues.

IMPACT:
- Application response times increased by 300%
- Some users may experience timeouts
- Data integrity remains intact

ACTIONS:
- Database team is investigating
- Backup systems are being prepared
- Customer support has been notified

Expected resolution: 45 minutes

For technical details, contact: tech-ops@company.com"""
    
    test_channels = [
        NotificationChannel.EMAIL,
        NotificationChannel.SMS,
        NotificationChannel.SLACK,
        NotificationChannel.MEDIA_RELEASE
    ]
    
    adaptations = await engine._adapt_content_for_channels(
        test_content, "Critical System Alert", test_channels
    )
    
    print(f"   📝 Original content: {len(test_content)} characters")
    print(f"   🔄 Adapted for {len(test_channels)} channels:")
    
    for channel_key, adaptation in adaptations.items():
        print(f"\n   📱 {channel_key.upper()}:")
        print(f"      Length: {len(adaptation['content'])} chars")
        print(f"      Truncated: {adaptation['metadata']['truncated']}")
        print(f"      Preview: {adaptation['content'][:80]}...")
    
    # Show coordination metrics
    print("\n10. System Coordination Metrics...")
    
    coord_metrics = engine.get_coordination_metrics()
    
    print(f"   📊 Overall Statistics:")
    print(f"      Total messages: {coord_metrics['total_messages']}")
    print(f"      Status distribution: {coord_metrics['status_distribution']}")
    print(f"      Channel usage: {coord_metrics['channel_usage']}")
    print(f"      Average approval time: {coord_metrics['average_approval_time']} minutes")
    
    effectiveness_summary = coord_metrics['effectiveness_summary']
    print(f"   📈 Effectiveness Summary:")
    print(f"      Average score: {effectiveness_summary['average_effectiveness_score']:.2f}")
    print(f"      Tracked messages: {effectiveness_summary['total_tracked_messages']}")
    print(f"      High performers: {effectiveness_summary['high_performing_messages']}")
    
    print("\n" + "="*80)
    print("MESSAGE COORDINATION DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✓ Multi-channel message coordination")
    print("✓ Channel-specific content adaptation")
    print("✓ Approval workflow management")
    print("✓ Version control and change tracking")
    print("✓ Message publication and delivery")
    print("✓ Effectiveness tracking and analytics")
    print("✓ Consistent messaging across channels")
    print("✓ Real-time status monitoring")


async def demonstrate_approval_scenarios():
    """Demonstrate different approval scenarios"""
    print("\n" + "-"*60)
    print("APPROVAL WORKFLOW SCENARIOS")
    print("-"*60)
    
    engine = MessageCoordinationEngine()
    
    # Scenario 1: Media release requiring multiple approvals
    print("\n1. Media Release Approval Scenario...")
    
    media_message = await engine.create_coordinated_message(
        crisis_id="crisis_media_001",
        message_type="media_statement",
        master_content="Official statement regarding recent service disruption.",
        master_subject="Official Company Statement",
        target_channels=[NotificationChannel.MEDIA_RELEASE, NotificationChannel.EMAIL],
        target_stakeholders=[StakeholderType.MEDIA, StakeholderType.PUBLIC],
        priority=NotificationPriority.HIGH,
        requires_approval=True,
        created_by="pr_manager"
    )
    
    print(f"   📋 Approval steps: {len(media_message.approval_workflow)}")
    for step in media_message.approval_workflow:
        print(f"      {step.step_order}. {step.approver_role}")
    
    # Scenario 2: Rejection and revision
    print("\n2. Message Rejection Scenario...")
    
    # Reject first approval
    first_step = media_message.approval_workflow[0]
    await engine.reject_message(
        media_message.id,
        "dept_head_001",
        first_step.approver_role,
        "Content needs to be more specific about the timeline"
    )
    
    print(f"   ❌ Message rejected by {first_step.approver_role}")
    print(f"   📝 Reason: {first_step.comments}")
    print(f"   📋 Status: {media_message.status.value}")
    
    # Scenario 3: Emergency bypass
    print("\n3. Emergency Message (No Approval) Scenario...")
    
    emergency_message = await engine.create_coordinated_message(
        crisis_id="crisis_emergency_001",
        message_type="emergency_alert",
        master_content="EMERGENCY: Evacuate building immediately due to fire alarm.",
        master_subject="EMERGENCY EVACUATION",
        target_channels=[NotificationChannel.SMS, NotificationChannel.PUSH],
        target_stakeholders=[StakeholderType.EMPLOYEE],
        priority=NotificationPriority.CRITICAL,
        requires_approval=False,  # Emergency bypass
        created_by="security_system"
    )
    
    print(f"   🚨 Emergency message created")
    print(f"   📋 Status: {emergency_message.status.value} (no approval required)")
    print(f"   ⚡ Ready for immediate publication")


if __name__ == "__main__":
    print("Starting Crisis Message Coordination System Demo...")
    
    # Run the main demonstration
    asyncio.run(demonstrate_message_coordination())
    
    # Run approval scenarios
    asyncio.run(demonstrate_approval_scenarios())
    
    print("\nDemo completed successfully! 🎉")
    print("\nThe message coordination system provides:")
    print("• Consistent messaging across all communication channels")
    print("• Channel-specific content adaptation and optimization")
    print("• Flexible approval workflows with role-based permissions")
    print("• Complete version control and change tracking")
    print("• Real-time publication and delivery coordination")
    print("• Comprehensive effectiveness tracking and analytics")
    print("• Emergency bypass capabilities for critical situations")
    print("• Scalable multi-channel message distribution")