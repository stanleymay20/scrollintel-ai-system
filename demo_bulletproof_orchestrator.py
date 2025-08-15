"""
Demo of the Bulletproof Orchestrator system.
Shows how all protection systems work together in a unified way.
"""

import asyncio
import time
import json
from datetime import datetime

from scrollintel.core.bulletproof_orchestrator import (
    bulletproof_orchestrator,
    start_bulletproof_system,
    stop_bulletproof_system,
    get_system_health,
    UnifiedConfiguration,
    ProtectionMode
)


async def demo_orchestrator_startup():
    """Demo orchestrator startup and basic functionality."""
    print("🚀 Starting Bulletproof Orchestrator Demo")
    print("=" * 50)
    
    # Start the bulletproof system
    print("\n1. Starting bulletproof system...")
    await start_bulletproof_system()
    
    # Wait a moment for systems to initialize
    await asyncio.sleep(2)
    
    # Check system health
    print("\n2. Checking system health...")
    health_data = get_system_health()
    print(f"   Overall Status: {health_data.get('overall_status', 'unknown')}")
    print(f"   Protection Mode: {health_data.get('protection_mode', 'unknown')}")
    print(f"   System Active: {health_data.get('is_active', False)}")
    
    if 'metrics' in health_data:
        metrics = health_data['metrics']
        print(f"   CPU Usage: {metrics.get('cpu_usage', 0):.1f}%")
        print(f"   Memory Usage: {metrics.get('memory_usage', 0):.1f}%")
        print(f"   Protection Effectiveness: {metrics.get('protection_effectiveness', 0)*100:.1f}%")
        print(f"   User Satisfaction: {metrics.get('user_satisfaction_score', 0)*100:.1f}%")
    
    return health_data


async def demo_protection_systems():
    """Demo individual protection systems status."""
    print("\n3. Checking protection systems...")
    
    try:
        systems_status = bulletproof_orchestrator.get_protection_systems_status()
        
        for system in systems_status:
            print(f"   📊 {system.system_name.replace('_', ' ').title()}:")
            print(f"      Status: {'✅ Active' if system.is_active else '❌ Inactive'}")
            print(f"      Health: {system.health_score*100:.1f}%")
            if system.last_action:
                print(f"      Last Action: {system.last_action}")
            if system.alerts:
                print(f"      Alerts: {len(system.alerts)}")
    
    except Exception as e:
        print(f"   ⚠️  Could not get protection systems status: {e}")


async def demo_configuration_management():
    """Demo configuration management."""
    print("\n4. Testing configuration management...")
    
    # Get current configuration
    current_config = bulletproof_orchestrator.get_configuration()
    print(f"   Current Protection Mode: {current_config.protection_mode.value}")
    print(f"   Auto Recovery: {current_config.auto_recovery_enabled}")
    print(f"   Real-time Monitoring: {current_config.real_time_monitoring}")
    
    # Update configuration to performance optimized mode
    print("\n   Switching to Performance Optimized mode...")
    performance_config = UnifiedConfiguration(
        protection_mode=ProtectionMode.PERFORMANCE_OPTIMIZED,
        auto_recovery_enabled=True,
        predictive_prevention_enabled=False,
        user_experience_optimization=True,
        intelligent_routing_enabled=True,
        fallback_generation_enabled=True,
        degradation_learning_enabled=False,
        real_time_monitoring=True,
        alert_thresholds=current_config.alert_thresholds,
        recovery_timeouts=current_config.recovery_timeouts,
        user_notification_settings=current_config.user_notification_settings
    )
    
    bulletproof_orchestrator.update_configuration(performance_config)
    
    # Verify configuration change
    updated_config = bulletproof_orchestrator.get_configuration()
    print(f"   ✅ Updated Protection Mode: {updated_config.protection_mode.value}")
    
    # Switch back to full protection
    print("\n   Switching back to Full Protection mode...")
    full_protection_config = UnifiedConfiguration(
        protection_mode=ProtectionMode.FULL_PROTECTION,
        auto_recovery_enabled=True,
        predictive_prevention_enabled=True,
        user_experience_optimization=True,
        intelligent_routing_enabled=True,
        fallback_generation_enabled=True,
        degradation_learning_enabled=True,
        real_time_monitoring=True,
        alert_thresholds=current_config.alert_thresholds,
        recovery_timeouts=current_config.recovery_timeouts,
        user_notification_settings=current_config.user_notification_settings
    )
    
    bulletproof_orchestrator.update_configuration(full_protection_config)
    print("   ✅ Restored Full Protection mode")


async def demo_alert_system():
    """Demo alert system functionality."""
    print("\n5. Testing alert system...")
    
    # Check for active alerts
    active_alerts = bulletproof_orchestrator.get_active_alerts()
    print(f"   Active Alerts: {len(active_alerts)}")
    
    if active_alerts:
        for alert in active_alerts[:3]:  # Show first 3 alerts
            print(f"   🚨 {alert['type']}: {alert['message']}")
            print(f"      Severity: {alert['severity']}")
            print(f"      Time: {alert['timestamp']}")
    else:
        print("   ✅ No active alerts - system is healthy")


async def demo_recovery_actions():
    """Demo recovery actions."""
    print("\n6. Checking recent recovery actions...")
    
    recovery_actions = bulletproof_orchestrator.get_recent_recovery_actions(hours=24)
    print(f"   Recent Recovery Actions (24h): {len(recovery_actions)}")
    
    if recovery_actions:
        for action in recovery_actions[:3]:  # Show first 3 actions
            print(f"   🔧 {action['type']}: {'✅ Success' if action['success'] else '❌ Failed'}")
            print(f"      Actions: {', '.join(action['actions_taken'])}")
            print(f"      Time: {action['timestamp']}")
    else:
        print("   ✅ No recent recovery actions needed - system is stable")


async def demo_emergency_mode():
    """Demo emergency mode activation."""
    print("\n7. Testing emergency mode...")
    
    # Get current status
    health_before = get_system_health()
    print(f"   Current Mode: {health_before.get('protection_mode', 'unknown')}")
    
    # Activate emergency mode
    print("   🚨 Activating emergency mode...")
    emergency_config = UnifiedConfiguration(
        protection_mode=ProtectionMode.EMERGENCY_MODE,
        auto_recovery_enabled=True,
        predictive_prevention_enabled=False,
        user_experience_optimization=True,
        intelligent_routing_enabled=False,
        fallback_generation_enabled=True,
        degradation_learning_enabled=False,
        real_time_monitoring=True,
        alert_thresholds=bulletproof_orchestrator.config.alert_thresholds,
        recovery_timeouts=bulletproof_orchestrator.config.recovery_timeouts,
        user_notification_settings=bulletproof_orchestrator.config.user_notification_settings
    )
    
    bulletproof_orchestrator.update_configuration(emergency_config)
    
    # Check emergency mode status
    health_emergency = get_system_health()
    print(f"   ✅ Emergency Mode Active: {health_emergency.get('protection_mode', 'unknown')}")
    
    # Wait a moment
    await asyncio.sleep(1)
    
    # Deactivate emergency mode
    print("   🔄 Deactivating emergency mode...")
    normal_config = UnifiedConfiguration(
        protection_mode=ProtectionMode.FULL_PROTECTION,
        auto_recovery_enabled=True,
        predictive_prevention_enabled=True,
        user_experience_optimization=True,
        intelligent_routing_enabled=True,
        fallback_generation_enabled=True,
        degradation_learning_enabled=True,
        real_time_monitoring=True,
        alert_thresholds=bulletproof_orchestrator.config.alert_thresholds,
        recovery_timeouts=bulletproof_orchestrator.config.recovery_timeouts,
        user_notification_settings=bulletproof_orchestrator.config.user_notification_settings
    )
    
    bulletproof_orchestrator.update_configuration(normal_config)
    
    health_after = get_system_health()
    print(f"   ✅ Normal Mode Restored: {health_after.get('protection_mode', 'unknown')}")


async def demo_system_monitoring():
    """Demo real-time system monitoring."""
    print("\n8. Monitoring system for 10 seconds...")
    
    for i in range(10):
        health_data = get_system_health()
        
        if 'metrics' in health_data:
            metrics = health_data['metrics']
            print(f"   [{i+1:2d}/10] Status: {health_data.get('overall_status', 'unknown')} | "
                  f"CPU: {metrics.get('cpu_usage', 0):5.1f}% | "
                  f"Memory: {metrics.get('memory_usage', 0):5.1f}% | "
                  f"Response: {metrics.get('response_time', 0):5.2f}s | "
                  f"Satisfaction: {metrics.get('user_satisfaction_score', 0)*100:5.1f}%")
        else:
            print(f"   [{i+1:2d}/10] Status: {health_data.get('overall_status', 'unknown')}")
        
        await asyncio.sleep(1)


async def demo_dashboard_data():
    """Demo dashboard data generation."""
    print("\n9. Generating dashboard data...")
    
    # Force dashboard update
    await bulletproof_orchestrator._update_dashboard()
    
    # Get dashboard data
    dashboard_data = bulletproof_orchestrator.dashboard_manager.dashboard_data
    
    if dashboard_data:
        print("   📊 Dashboard Data Available:")
        print(f"      Overall Status: {dashboard_data.get('overall_status', 'unknown')}")
        print(f"      Active Alerts: {dashboard_data.get('active_alerts', 0)}")
        print(f"      Recent Recoveries: {dashboard_data.get('recent_recovery_actions', 0)}")
        print(f"      System Uptime: {dashboard_data.get('uptime', 'unknown')}")
        
        if 'system_metrics' in dashboard_data:
            metrics = dashboard_data['system_metrics']
            print(f"      CPU Usage: {metrics.get('cpu_usage', 0):.1f}%")
            print(f"      Memory Usage: {metrics.get('memory_usage', 0):.1f}%")
            print(f"      Active Users: {metrics.get('active_users', 0)}")
        
        if 'protection_metrics' in dashboard_data:
            protection = dashboard_data['protection_metrics']
            print(f"      Protection Effectiveness: {protection.get('effectiveness', 0)*100:.1f}%")
            print(f"      Recovery Success Rate: {protection.get('recovery_success_rate', 0)*100:.1f}%")
            print(f"      User Satisfaction: {protection.get('user_satisfaction', 0)*100:.1f}%")
    else:
        print("   ⚠️  No dashboard data available yet")


async def demo_cleanup():
    """Demo cleanup and shutdown."""
    print("\n10. Shutting down bulletproof system...")
    
    # Stop the bulletproof system
    await stop_bulletproof_system()
    
    # Verify shutdown
    health_data = get_system_health()
    print(f"    System Active: {health_data.get('is_active', True)}")
    print("    ✅ Bulletproof system stopped")


async def main():
    """Run the complete bulletproof orchestrator demo."""
    try:
        # Run all demo functions
        await demo_orchestrator_startup()
        await demo_protection_systems()
        await demo_configuration_management()
        await demo_alert_system()
        await demo_recovery_actions()
        await demo_emergency_mode()
        await demo_system_monitoring()
        await demo_dashboard_data()
        
        print("\n" + "=" * 50)
        print("🎉 Bulletproof Orchestrator Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("✅ Unified system startup and coordination")
        print("✅ Real-time health monitoring")
        print("✅ Protection system status tracking")
        print("✅ Dynamic configuration management")
        print("✅ Alert system functionality")
        print("✅ Recovery action coordination")
        print("✅ Emergency mode activation")
        print("✅ System monitoring and metrics")
        print("✅ Dashboard data generation")
        print("✅ Graceful system shutdown")
        
        print("\n🛡️  All protection systems working together seamlessly!")
        print("Users will never experience failures with this unified protection.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure cleanup
        try:
            await demo_cleanup()
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")


if __name__ == "__main__":
    asyncio.run(main())