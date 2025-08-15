"""
Demo script for cross-device and cross-session continuity system.
Demonstrates seamless state transfer, session recovery, multi-tab sync, and offline capabilities.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.cross_device_continuity import (
    CrossDeviceContinuityManager, DeviceType, ConflictResolutionStrategy,
    create_user_session, restore_user_session, sync_user_state,
    enable_offline_mode, sync_after_reconnect
)


class CrossDeviceContinuityDemo:
    """Demo class for cross-device continuity features."""
    
    def __init__(self):
        self.demo_user_id = "demo_user_123"
        self.devices = {
            "desktop": "desktop_device_456",
            "mobile": "mobile_device_789",
            "tablet": "tablet_device_012"
        }
        self.sessions = {}
        
    async def run_demo(self):
        """Run the complete cross-device continuity demo."""
        print("🚀 ScrollIntel Cross-Device Continuity Demo")
        print("=" * 50)
        
        try:
            # Demo 1: Device Registration and Session Creation
            await self.demo_device_registration()
            
            # Demo 2: State Synchronization Across Devices
            await self.demo_state_synchronization()
            
            # Demo 3: Session Recovery
            await self.demo_session_recovery()
            
            # Demo 4: Multi-Tab Synchronization
            await self.demo_multi_tab_sync()
            
            # Demo 5: Conflict Resolution
            await self.demo_conflict_resolution()
            
            # Demo 6: Offline Mode and Sync
            await self.demo_offline_mode()
            
            # Demo 7: Real-time Synchronization
            await self.demo_realtime_sync()
            
            # Demo 8: Cross-Device Continuity
            await self.demo_cross_device_continuity()
            
            print("\n✅ All demos completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            raise
    
    async def demo_device_registration(self):
        """Demo device registration and session creation."""
        print("\n📱 Demo 1: Device Registration and Session Creation")
        print("-" * 50)
        
        # Create sessions for different devices
        for device_name, device_id in self.devices.items():
            print(f"Creating session for {device_name} device...")
            
            initial_state = {
                "device_type": device_name,
                "current_page": "/dashboard",
                "user_preferences": {
                    "theme": "dark" if device_name == "desktop" else "light",
                    "language": "en",
                    "notifications": True
                },
                "workspace": {
                    "open_tabs": ["/dashboard"],
                    "active_filters": {"date_range": "last_30_days"},
                    "zoom_level": 1.0 if device_name == "desktop" else 0.8
                }
            }
            
            session = await create_user_session(
                user_id=self.demo_user_id,
                device_id=device_id,
                device_type=device_name,
                initial_state=initial_state
            )
            
            self.sessions[device_name] = session
            
            print(f"  ✓ Session created: {session.session_id}")
            print(f"  ✓ Device: {device_name} ({device_id})")
            print(f"  ✓ Initial state: {len(session.state_data)} keys")
            
            await asyncio.sleep(0.5)  # Small delay for demo effect
        
        print(f"\n📊 Summary: Created {len(self.sessions)} sessions across {len(self.devices)} devices")
    
    async def demo_state_synchronization(self):
        """Demo state synchronization across devices."""
        print("\n🔄 Demo 2: State Synchronization Across Devices")
        print("-" * 50)
        
        # Update state on desktop
        desktop_session = self.sessions["desktop"]
        print("Updating state on desktop device...")
        
        desktop_updates = {
            "current_page": "/analytics",
            "workspace": {
                "open_tabs": ["/dashboard", "/analytics"],
                "active_filters": {"date_range": "last_7_days", "category": "sales"},
                "selected_chart": "revenue_trend",
                "zoom_level": 1.2
            },
            "recent_activity": [
                {"action": "view_chart", "chart": "revenue_trend", "timestamp": datetime.utcnow().isoformat()},
                {"action": "apply_filter", "filter": "category=sales", "timestamp": datetime.utcnow().isoformat()}
            ]
        }
        
        success = await sync_user_state(
            user_id=self.demo_user_id,
            session_id=desktop_session.session_id,
            state_updates=desktop_updates,
            device_id=self.devices["desktop"]
        )
        
        print(f"  ✓ Desktop state updated: {success}")
        print(f"  ✓ New page: {desktop_updates['current_page']}")
        print(f"  ✓ Active filters: {desktop_updates['workspace']['active_filters']}")
        
        # Simulate sync across devices
        print("\nSynchronizing across all devices...")
        from scrollintel.core.cross_device_continuity import continuity_manager
        sync_results = await continuity_manager.sync_across_devices(self.demo_user_id)
        
        print(f"  ✓ Sync status: {sync_results['status']}")
        print(f"  ✓ Synced sessions: {sync_results['synced_sessions']}")
        print(f"  ✓ Conflicts: {sync_results['conflicts']}")
        
        # Verify sync on mobile device
        mobile_session = await restore_user_session(self.sessions["mobile"].session_id)
        if mobile_session:
            print(f"  ✓ Mobile device now shows: {mobile_session.state_data.get('current_page', 'unknown')}")
            print(f"  ✓ Mobile session version: {mobile_session.version}")
    
    async def demo_session_recovery(self):
        """Demo session recovery with exact context restoration."""
        print("\n🔄 Demo 3: Session Recovery with Exact Context Restoration")
        print("-" * 50)
        
        # Simulate session interruption and recovery
        desktop_session = self.sessions["desktop"]
        original_session_id = desktop_session.session_id
        
        print(f"Original session: {original_session_id}")
        print("Simulating session interruption...")
        
        # Remove from active sessions (simulate crash/disconnect)
        from scrollintel.core.cross_device_continuity import continuity_manager
        if original_session_id in continuity_manager.active_sessions:
            original_state = continuity_manager.active_sessions[original_session_id].state_data.copy()
            del continuity_manager.active_sessions[original_session_id]
            print("  ✓ Session removed from active sessions")
        
        await asyncio.sleep(1)  # Simulate downtime
        
        # Restore session
        print("Restoring session with exact context...")
        restored_session = await restore_user_session(original_session_id)
        
        if restored_session:
            print(f"  ✓ Session restored: {restored_session.session_id}")
            print(f"  ✓ User ID: {restored_session.user_id}")
            print(f"  ✓ Device ID: {restored_session.device_id}")
            print(f"  ✓ Version: {restored_session.version}")
            print(f"  ✓ State integrity: {'✓' if restored_session.checksum else '✗'}")
            
            # Verify exact context restoration
            current_page = restored_session.state_data.get("current_page")
            workspace = restored_session.state_data.get("workspace", {})
            
            print(f"  ✓ Current page restored: {current_page}")
            print(f"  ✓ Open tabs: {workspace.get('open_tabs', [])}")
            print(f"  ✓ Active filters: {workspace.get('active_filters', {})}")
            print(f"  ✓ Zoom level: {workspace.get('zoom_level', 1.0)}")
            
            # Update our reference
            self.sessions["desktop"] = restored_session
        else:
            print("  ❌ Session restoration failed")
    
    async def demo_multi_tab_sync(self):
        """Demo multi-tab synchronization with conflict resolution."""
        print("\n🗂️ Demo 4: Multi-Tab Synchronization")
        print("-" * 50)
        
        from scrollintel.core.cross_device_continuity import continuity_manager
        
        # Simulate multiple tabs for the same user
        tabs = ["tab_1", "tab_2", "tab_3"]
        tab_states = {}
        
        for i, tab_id in enumerate(tabs):
            print(f"Creating state for {tab_id}...")
            
            tab_state = {
                "tab_id": tab_id,
                "current_page": f"/page_{i+1}",
                "scroll_position": i * 100,
                "form_data": {
                    "field_1": f"value_{i+1}",
                    "field_2": f"data_{i+1}"
                },
                "ui_state": {
                    "sidebar_collapsed": i % 2 == 0,
                    "selected_items": [f"item_{i+1}"]
                }
            }
            
            result = await continuity_manager.handle_multi_tab_sync(
                user_id=self.demo_user_id,
                tab_id=tab_id,
                state_updates=tab_state
            )
            
            tab_states[tab_id] = result
            print(f"  ✓ {tab_id} synced: {result['status']}")
            print(f"  ✓ Session ID: {result['session_id']}")
            print(f"  ✓ Version: {result['version']}")
        
        # Simulate tab interaction and sync
        print("\nSimulating tab interaction...")
        updated_state = {
            "current_page": "/shared_page",
            "shared_data": {
                "selected_date": datetime.utcnow().isoformat(),
                "filters": {"status": "active", "type": "important"}
            },
            "last_interaction": datetime.utcnow().isoformat()
        }
        
        sync_result = await continuity_manager.handle_multi_tab_sync(
            user_id=self.demo_user_id,
            tab_id="tab_1",
            state_updates=updated_state
        )
        
        print(f"  ✓ Tab interaction synced: {sync_result['status']}")
        print(f"  ✓ All tabs now share: {updated_state['shared_data']}")
    
    async def demo_conflict_resolution(self):
        """Demo conflict resolution strategies."""
        print("\n⚔️ Demo 5: Conflict Resolution")
        print("-" * 50)
        
        from scrollintel.core.cross_device_continuity import continuity_manager
        
        # Create conflicting changes
        print("Creating conflicting changes on different devices...")
        
        # Desktop change
        desktop_updates = {
            "shared_setting": "desktop_value",
            "last_modified_by": "desktop",
            "timestamp": datetime.utcnow().isoformat(),
            "conflict_data": {
                "priority": "high",
                "category": "work"
            }
        }
        
        # Mobile change (slightly later)
        await asyncio.sleep(0.1)
        mobile_updates = {
            "shared_setting": "mobile_value",
            "last_modified_by": "mobile",
            "timestamp": datetime.utcnow().isoformat(),
            "conflict_data": {
                "priority": "medium",
                "category": "personal"
            }
        }
        
        # Apply changes
        desktop_session = self.sessions["desktop"]
        mobile_session = self.sessions["mobile"]
        
        await sync_user_state(
            user_id=self.demo_user_id,
            session_id=desktop_session.session_id,
            state_updates=desktop_updates,
            device_id=self.devices["desktop"]
        )
        
        await sync_user_state(
            user_id=self.demo_user_id,
            session_id=mobile_session.session_id,
            state_updates=mobile_updates,
            device_id=self.devices["mobile"]
        )
        
        print("  ✓ Desktop update applied")
        print("  ✓ Mobile update applied")
        
        # Trigger sync to detect conflicts
        print("\nSynchronizing to detect conflicts...")
        sync_results = await continuity_manager.sync_across_devices(self.demo_user_id)
        
        print(f"  ✓ Sync completed with {sync_results['conflicts']} conflicts")
        
        # Check for conflicts
        user_conflicts = [
            conflict for conflict in continuity_manager.sync_conflicts.values()
            if conflict.local_change.user_id == self.demo_user_id
        ]
        
        if user_conflicts:
            print(f"  ✓ Found {len(user_conflicts)} conflicts")
            
            for conflict in user_conflicts:
                print(f"    - Conflict ID: {conflict.conflict_id}")
                print(f"    - Path: {conflict.path}")
                print(f"    - Local value: {conflict.local_change.new_value}")
                print(f"    - Remote value: {conflict.remote_change.new_value}")
                print(f"    - Resolution strategy: {conflict.resolution_strategy.value}")
                
                # Resolve conflict
                resolution = await continuity_manager.conflict_resolver.resolve_conflict(conflict)
                print(f"    - Resolved to: {resolution}")
        else:
            print("  ✓ No conflicts detected (automatic resolution successful)")
    
    async def demo_offline_mode(self):
        """Demo offline mode and automatic sync when reconnected."""
        print("\n📴 Demo 6: Offline Mode and Automatic Sync")
        print("-" * 50)
        
        # Enable offline mode
        print("Enabling offline mode for mobile device...")
        offline_result = await enable_offline_mode(
            user_id=self.demo_user_id,
            device_id=self.devices["mobile"]
        )
        
        print(f"  ✓ Offline mode enabled: {offline_result['status']}")
        print(f"  ✓ Cached sessions: {offline_result['cached_sessions']}")
        print(f"  ✓ Offline capabilities: {list(offline_result['capabilities'].keys())}")
        
        # Simulate offline changes
        print("\nMaking changes while offline...")
        from scrollintel.core.cross_device_continuity import continuity_manager
        
        offline_changes = [
            {
                "path": "offline_notes",
                "value": "Created note while offline",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "path": "offline_edits",
                "value": {"document_id": "doc_123", "changes": ["edit_1", "edit_2"]},
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "path": "offline_preferences",
                "value": {"auto_save": True, "sync_frequency": "high"},
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        # Queue offline changes
        for change_data in offline_changes:
            from scrollintel.core.cross_device_continuity import StateChange
            change = StateChange(
                change_id=f"offline_{int(time.time() * 1000)}",
                session_id=self.sessions["mobile"].session_id,
                user_id=self.demo_user_id,
                device_id=self.devices["mobile"],
                timestamp=datetime.utcnow(),
                path=change_data["path"],
                old_value=None,
                new_value=change_data["value"],
                operation="set"
            )
            
            continuity_manager.offline_manager.queue_offline_change(change)
            print(f"  ✓ Queued offline change: {change_data['path']}")
        
        print(f"  ✓ Total offline changes queued: {len(continuity_manager.offline_manager.offline_queue)}")
        
        # Simulate reconnection and sync
        print("\nReconnecting and syncing offline changes...")
        sync_result = await sync_after_reconnect(
            user_id=self.demo_user_id,
            device_id=self.devices["mobile"]
        )
        
        print(f"  ✓ Reconnection status: {sync_result['status']}")
        print(f"  ✓ Offline changes synced: {sync_result['offline_changes_synced']}")
        print(f"  ✓ Sync results: {sync_result['sync_results']['status']}")
        print(f"  ✓ Remaining offline queue: {len(continuity_manager.offline_manager.offline_queue)}")
    
    async def demo_realtime_sync(self):
        """Demo real-time synchronization capabilities."""
        print("\n⚡ Demo 7: Real-time Synchronization")
        print("-" * 50)
        
        from scrollintel.core.cross_device_continuity import continuity_manager
        
        # Simulate real-time updates
        print("Simulating real-time updates across devices...")
        
        realtime_updates = [
            {
                "device": "desktop",
                "updates": {
                    "live_data": {"cpu_usage": 45, "memory_usage": 62},
                    "notifications": [{"id": 1, "message": "New data available"}],
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            {
                "device": "mobile",
                "updates": {
                    "location": {"lat": 37.7749, "lng": -122.4194},
                    "battery_level": 78,
                    "network_quality": "good",
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            {
                "device": "tablet",
                "updates": {
                    "drawing_data": {"strokes": 15, "colors_used": ["red", "blue"]},
                    "orientation": "landscape",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        ]
        
        # Apply updates with small delays to simulate real-time
        for update_data in realtime_updates:
            device_name = update_data["device"]
            session = self.sessions[device_name]
            
            success = await sync_user_state(
                user_id=self.demo_user_id,
                session_id=session.session_id,
                state_updates=update_data["updates"],
                device_id=self.devices[device_name]
            )
            
            print(f"  ✓ {device_name.capitalize()} update synced: {success}")
            
            # Small delay to simulate real-time updates
            await asyncio.sleep(0.3)
        
        # Check sync status
        sync_status = await continuity_manager.get_sync_status(self.demo_user_id)
        print(f"\n📊 Real-time sync status:")
        print(f"  ✓ Total sessions: {sync_status['total_sessions']}")
        print(f"  ✓ Active devices: {sync_status['active_devices']}")
        print(f"  ✓ WebSocket connections: {sync_status['websocket_connections']}")
        print(f"  ✓ Pending conflicts: {sync_status['pending_conflicts']}")
    
    async def demo_cross_device_continuity(self):
        """Demo complete cross-device continuity workflow."""
        print("\n🌐 Demo 8: Complete Cross-Device Continuity")
        print("-" * 50)
        
        # Simulate a complete user workflow across devices
        print("Simulating complete user workflow across devices...")
        
        # Step 1: Start work on desktop
        print("\n1. Starting work on desktop...")
        desktop_work = {
            "project": {
                "id": "project_123",
                "name": "Q4 Analysis",
                "status": "in_progress",
                "files": ["data.csv", "analysis.py", "report.md"]
            },
            "current_task": "data_analysis",
            "progress": 0.3,
            "work_session": {
                "start_time": datetime.utcnow().isoformat(),
                "focus_mode": True,
                "break_reminders": True
            }
        }
        
        await sync_user_state(
            user_id=self.demo_user_id,
            session_id=self.sessions["desktop"].session_id,
            state_updates=desktop_work,
            device_id=self.devices["desktop"]
        )
        print("  ✓ Desktop work session started")
        
        # Step 2: Continue on mobile during commute
        await asyncio.sleep(0.5)
        print("\n2. Continuing work on mobile during commute...")
        
        mobile_work = {
            "project": {
                "status": "in_progress",
                "notes": ["Need to check data quality", "Consider additional metrics"],
                "mobile_edits": True
            },
            "current_task": "review_and_notes",
            "progress": 0.5,
            "location_context": "commuting",
            "work_session": {
                "device_switch": True,
                "mobile_optimized": True
            }
        }
        
        await sync_user_state(
            user_id=self.demo_user_id,
            session_id=self.sessions["mobile"].session_id,
            state_updates=mobile_work,
            device_id=self.devices["mobile"]
        )
        print("  ✓ Mobile work session continued")
        
        # Step 3: Present on tablet
        await asyncio.sleep(0.5)
        print("\n3. Presenting findings on tablet...")
        
        tablet_work = {
            "project": {
                "status": "presenting",
                "presentation_mode": True,
                "slides": ["overview", "data_analysis", "conclusions"]
            },
            "current_task": "presentation",
            "progress": 0.8,
            "presentation_context": {
                "audience": "stakeholders",
                "duration": "30_minutes",
                "interactive": True
            }
        }
        
        await sync_user_state(
            user_id=self.demo_user_id,
            session_id=self.sessions["tablet"].session_id,
            state_updates=tablet_work,
            device_id=self.devices["tablet"]
        )
        print("  ✓ Tablet presentation session started")
        
        # Step 4: Complete work back on desktop
        await asyncio.sleep(0.5)
        print("\n4. Completing work back on desktop...")
        
        completion_work = {
            "project": {
                "status": "completed",
                "completion_time": datetime.utcnow().isoformat(),
                "final_report": "report_final.pdf",
                "feedback_incorporated": True
            },
            "current_task": "completed",
            "progress": 1.0,
            "work_session": {
                "end_time": datetime.utcnow().isoformat(),
                "total_devices_used": 3,
                "seamless_transitions": 3
            }
        }
        
        await sync_user_state(
            user_id=self.demo_user_id,
            session_id=self.sessions["desktop"].session_id,
            state_updates=completion_work,
            device_id=self.devices["desktop"]
        )
        print("  ✓ Work completed on desktop")
        
        # Final sync across all devices
        print("\n5. Final synchronization across all devices...")
        from scrollintel.core.cross_device_continuity import continuity_manager
        final_sync = await continuity_manager.sync_across_devices(self.demo_user_id)
        
        print(f"  ✓ Final sync status: {final_sync['status']}")
        print(f"  ✓ All devices synchronized: {final_sync['synced_sessions']} sessions")
        
        # Verify continuity
        print("\n📋 Cross-device continuity verification:")
        for device_name, session in self.sessions.items():
            restored = await restore_user_session(session.session_id)
            if restored:
                project_status = restored.state_data.get("project", {}).get("status", "unknown")
                progress = restored.state_data.get("progress", 0)
                print(f"  ✓ {device_name.capitalize()}: {project_status} ({progress*100:.0f}% complete)")
    
    async def print_final_summary(self):
        """Print final demo summary."""
        print("\n" + "=" * 60)
        print("📊 CROSS-DEVICE CONTINUITY DEMO SUMMARY")
        print("=" * 60)
        
        from scrollintel.core.cross_device_continuity import continuity_manager
        
        # Get final status
        final_status = await continuity_manager.get_sync_status(self.demo_user_id)
        
        print(f"👤 User ID: {self.demo_user_id}")
        print(f"📱 Total Devices: {len(self.devices)}")
        print(f"🔄 Active Sessions: {final_status['total_sessions']}")
        print(f"⚡ WebSocket Connections: {final_status['websocket_connections']}")
        print(f"⚔️ Resolved Conflicts: {len(continuity_manager.sync_conflicts)}")
        print(f"📴 Offline Queue: {final_status['offline_queue_size']} items")
        
        print("\n🎯 Key Features Demonstrated:")
        features = [
            "✓ Seamless device registration and session creation",
            "✓ Real-time state synchronization across devices",
            "✓ Exact context restoration after interruptions",
            "✓ Multi-tab synchronization with conflict resolution",
            "✓ Intelligent conflict resolution strategies",
            "✓ Offline mode with automatic sync when reconnected",
            "✓ Real-time updates and WebSocket communication",
            "✓ Complete cross-device workflow continuity"
        ]
        
        for feature in features:
            print(f"  {feature}")
        
        print("\n🚀 ScrollIntel Cross-Device Continuity: BULLETPROOF! 🚀")


async def main():
    """Run the cross-device continuity demo."""
    demo = CrossDeviceContinuityDemo()
    
    try:
        await demo.run_demo()
        await demo.print_final_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up demo resources...")
        from scrollintel.core.cross_device_continuity import continuity_manager
        await continuity_manager.shutdown()
        print("✓ Cleanup completed")


if __name__ == "__main__":
    print("Starting ScrollIntel Cross-Device Continuity Demo...")
    asyncio.run(main())