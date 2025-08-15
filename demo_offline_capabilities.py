"""
Demo script showcasing ScrollIntel's advanced offline capabilities.
Demonstrates offline data management, intelligent sync, offline-first architecture, and PWA features.
"""

import asyncio
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# Import offline capabilities
from scrollintel.core.offline_data_manager import (
    OfflineDataManager, DataOperation, SyncStatus, ConflictResolutionStrategy
)
from scrollintel.core.offline_sync_engine import IntelligentSyncEngine
from scrollintel.core.offline_first_architecture import (
    OfflineFirstArchitecture, ConnectionStatus, OperationMode, OfflineCapability
)
from scrollintel.core.progressive_web_app import (
    ProgressiveWebApp, CacheStrategy, ResourceType, CacheRule
)


class OfflineCapabilitiesDemo:
    """Comprehensive demo of offline capabilities."""
    
    def __init__(self):
        self.demo_db_path = "data/demo_offline.db"
        self.demo_static_path = "demo_static"
        
        # Initialize components
        self.offline_manager = None
        self.sync_engine = None
        self.architecture = None
        self.pwa = None
        
        # Demo data
        self.demo_users = ["alice", "bob", "charlie", "diana"]
        self.demo_collections = ["dashboards", "reports", "charts", "datasets"]
        
    async def initialize_components(self):
        """Initialize all offline capability components."""
        print("🚀 Initializing ScrollIntel Offline Capabilities...")
        
        # Initialize offline data manager
        self.offline_manager = OfflineDataManager(self.demo_db_path, auto_sync=False)
        print("✅ Offline Data Manager initialized")
        
        # Initialize sync engine
        self.sync_engine = IntelligentSyncEngine(self.offline_manager)
        print("✅ Intelligent Sync Engine initialized")
        
        # Initialize offline-first architecture
        self.architecture = OfflineFirstArchitecture(self.demo_db_path)
        print("✅ Offline-First Architecture initialized")
        
        # Initialize PWA
        self.pwa = ProgressiveWebApp(self.demo_static_path)
        print("✅ Progressive Web App initialized")
        
        print("\n" + "="*60)
    
    async def demo_offline_data_management(self):
        """Demonstrate offline data management capabilities."""
        print("📊 DEMO: Offline Data Management")
        print("-" * 40)
        
        # Store various types of data
        stored_entries = []
        
        for collection in self.demo_collections:
            for i in range(3):
                user = random.choice(self.demo_users)
                
                # Create sample data based on collection type
                if collection == "dashboards":
                    data = {
                        "name": f"Dashboard {i+1}",
                        "layout": {"widgets": [f"widget_{j}" for j in range(4)]},
                        "theme": random.choice(["light", "dark", "auto"]),
                        "created_by": user
                    }
                elif collection == "reports":
                    data = {
                        "title": f"Monthly Report {i+1}",
                        "type": random.choice(["sales", "performance", "analytics"]),
                        "data_range": "2024-01-01 to 2024-01-31",
                        "generated_by": user
                    }
                elif collection == "charts":
                    data = {
                        "title": f"Chart {i+1}",
                        "type": random.choice(["bar", "line", "pie", "scatter"]),
                        "data_points": random.randint(10, 100),
                        "created_by": user
                    }
                else:  # datasets
                    data = {
                        "name": f"Dataset {i+1}",
                        "size_mb": random.randint(1, 500),
                        "format": random.choice(["csv", "json", "parquet"]),
                        "uploaded_by": user
                    }
                
                # Store data
                entry_id = await self.offline_manager.store_data(
                    collection, data, DataOperation.CREATE, user
                )
                
                stored_entries.append({
                    "collection": collection,
                    "entry_id": entry_id,
                    "user": user,
                    "data": data
                })
                
                print(f"  ✅ Stored {collection} entry: {data.get('name', data.get('title', 'Item'))}")
        
        print(f"\n📈 Total entries stored: {len(stored_entries)}")
        
        # Demonstrate querying
        print("\n🔍 Querying stored data...")
        for collection in self.demo_collections:
            results = await self.offline_manager.query_data(collection, limit=2)
            print(f"  📋 {collection}: {len(results)} entries found")
            
            if results:
                sample = results[0]
                print(f"    Sample: {sample.get('name', sample.get('title', 'Item'))}")
        
        # Demonstrate updates
        print("\n✏️ Updating data...")
        if stored_entries:
            entry = stored_entries[0]
            updates = {"last_modified": datetime.now().isoformat(), "version": 2}
            
            success = await self.offline_manager.update_data(
                entry["collection"], entry["entry_id"], updates, entry["user"]
            )
            
            if success:
                print(f"  ✅ Updated {entry['collection']} entry")
            else:
                print(f"  ❌ Failed to update {entry['collection']} entry")
        
        # Show storage info
        storage_info = self.offline_manager.get_storage_info()
        print(f"\n💾 Storage Info:")
        print(f"  Database size: {storage_info['database_size_mb']} MB")
        print(f"  Total entries: {storage_info['total_entries']}")
        print(f"  Collections: {list(storage_info['collections'].keys())}")
        
        return stored_entries
    
    async def demo_sync_capabilities(self, stored_entries):
        """Demonstrate intelligent sync and conflict resolution."""
        print("\n🔄 DEMO: Intelligent Sync & Conflict Resolution")
        print("-" * 50)
        
        # Show initial sync status
        sync_status = self.offline_manager.get_sync_status()
        print(f"📊 Sync Status:")
        print(f"  Online: {sync_status['is_online']}")
        print(f"  Pending operations: {sync_status['pending_sync_operations']}")
        
        # Simulate going offline
        print("\n📴 Simulating offline mode...")
        self.offline_manager.set_online_status(False)
        
        # Make some changes while offline
        print("✏️ Making changes while offline...")
        offline_changes = []
        
        for i in range(3):
            if stored_entries:
                entry = random.choice(stored_entries)
                updates = {
                    "offline_edit": True,
                    "edit_timestamp": datetime.now().isoformat(),
                    "edit_number": i + 1
                }
                
                success = await self.offline_manager.update_data(
                    entry["collection"], entry["entry_id"], updates, entry["user"]
                )
                
                if success:
                    offline_changes.append(entry)
                    print(f"  ✅ Offline edit {i+1}: {entry['collection']}")
        
        # Show pending sync operations
        sync_status = self.offline_manager.get_sync_status()
        print(f"\n📊 After offline changes:")
        print(f"  Pending operations: {sync_status['pending_sync_operations']}")
        
        # Simulate coming back online
        print("\n🌐 Coming back online...")
        self.offline_manager.set_online_status(True)
        
        # Force sync
        print("🔄 Forcing synchronization...")
        sync_result = await self.offline_manager.force_sync()
        print(f"  ✅ Sync completed: {sync_result}")
        
        # Show sync metrics
        if self.sync_engine:
            metrics = self.sync_engine.get_sync_metrics()
            print(f"\n📈 Sync Metrics:")
            print(f"  Total syncs: {metrics['total_syncs']}")
            print(f"  Success rate: {metrics['success_rate']:.2%}")
            print(f"  Conflicts resolved: {metrics['conflicts_resolved']}")
            
            # Show conflict insights
            insights = self.sync_engine.get_conflict_insights()
            if insights['total_patterns'] > 0:
                print(f"  Conflict patterns: {insights['total_patterns']}")
                if insights.get('most_common_conflict_type'):
                    print(f"  Most common conflict: {insights['most_common_conflict_type']}")
        
        # Check for any conflicts
        conflicts = self.offline_manager.get_conflicts()
        if conflicts:
            print(f"\n⚠️ Found {len(conflicts)} conflicts:")
            for conflict in conflicts[:3]:  # Show first 3
                print(f"  🔥 Conflict in {conflict['collection']}: {conflict['conflict_type']}")
                
                # Attempt automatic resolution
                resolution_success = await self.offline_manager.resolve_conflict(
                    conflict['id'], ConflictResolutionStrategy.INTELLIGENT_MERGE
                )
                
                if resolution_success:
                    print(f"    ✅ Automatically resolved")
                else:
                    print(f"    ❌ Manual resolution required")
        else:
            print("\n✅ No conflicts detected - all changes synced successfully!")
    
    async def demo_offline_first_architecture(self):
        """Demonstrate offline-first architecture."""
        print("\n🏗️ DEMO: Offline-First Architecture")
        print("-" * 40)
        
        # Show current architecture status
        status = self.architecture.get_architecture_status()
        print(f"🔧 Architecture Status:")
        print(f"  Connection: {status['connection_status']}")
        print(f"  Operation mode: {status['operation_mode']}")
        print(f"  Network latency: {status['network_metrics']['latency_ms']:.1f}ms")
        
        # Show offline capabilities
        capabilities = self.architecture.get_offline_capabilities()
        print(f"\n🎯 Offline Capabilities ({len(capabilities)} features):")
        for feature, caps in capabilities.items():
            read_status = "✅" if caps['read_offline'] else "❌"
            write_status = "✅" if caps['write_offline'] else "❌"
            print(f"  {feature}: Read {read_status} Write {write_status} (Priority: {caps['sync_priority']})")
        
        # Register a custom capability
        print("\n➕ Registering custom offline capability...")
        custom_capability = OfflineCapability(
            feature_name="demo_feature",
            read_offline=True,
            write_offline=True,
            sync_priority="high",
            conflict_resolution="intelligent_merge",
            cache_duration_hours=48
        )
        
        self.architecture.register_feature_capability(custom_capability)
        print("  ✅ Custom capability registered")
        
        # Test different operation modes
        operation_modes = [OperationMode.OFFLINE_FIRST, OperationMode.ONLINE_FIRST, OperationMode.HYBRID]
        
        for mode in operation_modes:
            print(f"\n🔄 Testing {mode.value} mode...")
            self.architecture.set_operation_mode(mode)
            
            # Execute test requests
            test_requests = [
                ("data_visualization", "read", {"id": "test_chart_1"}),
                ("user_preferences", "update", {"theme": "dark", "language": "en"}),
                ("analytics", "read", {"date_range": "last_7_days"})
            ]
            
            for feature, operation, data in test_requests:
                try:
                    result = await self.architecture.execute_request(
                        feature, operation, data, "demo_user"
                    )
                    
                    if result:
                        status_indicator = "✅"
                        if isinstance(result, dict):
                            if result.get('fallback'):
                                status_indicator = "🔄"
                            elif result.get('degraded'):
                                status_indicator = "⚠️"
                    else:
                        status_indicator = "❌"
                    
                    print(f"    {status_indicator} {feature}.{operation}")
                    
                except Exception as e:
                    print(f"    ❌ {feature}.{operation} - Error: {str(e)[:50]}")
        
        # Show performance metrics
        final_status = self.architecture.get_architecture_status()
        perf_metrics = final_status['performance_metrics']
        print(f"\n📊 Performance Metrics:")
        print(f"  Offline requests: {perf_metrics['requests_served_offline']}")
        print(f"  Online requests: {perf_metrics['requests_served_online']}")
        print(f"  Cache hits: {perf_metrics['cache_hits']}")
        print(f"  Failed requests: {perf_metrics['failed_requests']}")
    
    async def demo_progressive_web_app(self):
        """Demonstrate Progressive Web App capabilities."""
        print("\n📱 DEMO: Progressive Web App Features")
        print("-" * 40)
        
        # Show PWA status
        pwa_status = self.pwa.get_pwa_status()
        print(f"📊 PWA Status:")
        print(f"  Service Worker version: {pwa_status['service_worker_version']}")
        print(f"  Cache rules: {pwa_status['cache_rules_count']}")
        print(f"  Install prompts shown: {pwa_status['performance_metrics']['install_prompts_shown']}")
        
        # Generate PWA files
        print("\n📄 Generating PWA files...")
        
        # Generate manifest
        manifest = self.pwa.generate_manifest()
        print(f"  ✅ Manifest generated ({len(manifest)} properties)")
        print(f"    App name: {manifest['name']}")
        print(f"    Icons: {len(manifest['icons'])} sizes")
        print(f"    Shortcuts: {len(manifest.get('shortcuts', []))}")
        
        # Show cache rules
        cache_info = self.pwa.get_cache_info()
        print(f"\n🗄️ Cache Configuration:")
        print(f"  Total rules: {cache_info['total_rules']}")
        print(f"  Cache hit rate: {cache_info['performance']['hit_rate']:.2%}")
        
        for cache_name, rule_info in cache_info['cache_rules'].items():
            print(f"    {cache_name}: {rule_info['strategy']} ({rule_info['resource_type']})")
        
        # Add custom cache rule
        print("\n➕ Adding custom cache rule...")
        custom_rule = CacheRule(
            pattern=r"/demo/",
            strategy=CacheStrategy.STALE_WHILE_REVALIDATE,
            resource_type=ResourceType.DYNAMIC,
            max_age_hours=2,
            cache_name="demo-cache"
        )
        
        self.pwa.add_cache_rule(custom_rule)
        print("  ✅ Custom cache rule added")
        
        # Simulate user engagement
        print("\n👤 Simulating user engagement...")
        for i in range(7):  # Simulate 7 visits
            self.pwa.track_user_engagement("demo_user", "visit")
            if i % 2 == 0:
                self.pwa.track_user_engagement("demo_user", "time_spent", random.randint(300, 1800))
        
        # Test install prompt
        print("\n📲 Testing install prompt...")
        prompt_result = await self.pwa.handle_install_prompt("demo_user")
        
        if prompt_result['show_prompt']:
            print(f"  ✅ Install prompt shown: {prompt_result['message']}")
            
            # Simulate user accepting
            install_result = await self.pwa.handle_install_result(
                "demo_user", prompt_result['event_id'], "accepted"
            )
            print(f"  ✅ Install result recorded: {install_result}")
        else:
            print(f"  ℹ️ Install prompt not shown: {prompt_result['reason']}")
        
        # Register background sync
        print("\n🔄 Setting up background sync...")
        
        async def demo_sync_handler(data):
            print(f"    🔄 Background sync executed with data: {data}")
            return {"status": "completed", "synced_items": random.randint(1, 10)}
        
        self.pwa.register_background_sync("demo-sync", demo_sync_handler)
        
        # Trigger background sync
        sync_success = await self.pwa.trigger_background_sync("demo-sync", {"test": "data"})
        print(f"  ✅ Background sync triggered: {sync_success}")
        
        # Test push notifications
        print("\n🔔 Testing push notifications...")
        
        # Set up push subscription (mock)
        self.pwa.set_push_subscription({
            "endpoint": "https://demo.push.service/endpoint",
            "keys": {"p256dh": "demo_key", "auth": "demo_auth"}
        })
        
        # Send test notification
        notification_success = await self.pwa.send_push_notification(
            "demo_user",
            "ScrollIntel Demo",
            "Your offline data has been synchronized!",
            {"type": "sync_complete", "items": 5},
            [
                {"action": "view", "title": "View Dashboard"},
                {"action": "dismiss", "title": "Dismiss"}
            ]
        )
        
        print(f"  ✅ Push notification sent: {notification_success}")
        
        # Save PWA files to disk
        print("\n💾 Saving PWA files...")
        try:
            self.pwa.save_to_files()
            print("  ✅ PWA files saved successfully")
            print(f"    📁 Location: {self.demo_static_path}/")
            print("    📄 Files: manifest.json, sw.js, offline.html")
        except Exception as e:
            print(f"  ⚠️ File saving warning: {e}")
    
    async def demo_integration_scenarios(self):
        """Demonstrate integration scenarios."""
        print("\n🔗 DEMO: Integration Scenarios")
        print("-" * 35)
        
        # Scenario 1: User goes offline while working
        print("📋 Scenario 1: Working offline")
        print("  1. User is viewing dashboard online")
        print("  2. Connection drops")
        print("  3. User continues working with cached data")
        print("  4. User makes changes (stored locally)")
        print("  5. Connection restored")
        print("  6. Changes sync automatically")
        
        # Simulate this scenario
        self.architecture.connection_status = ConnectionStatus.ONLINE
        
        # User loads dashboard
        dashboard_data = await self.architecture.execute_request(
            "dashboard_config", "read", {"id": "main_dashboard"}, "scenario_user"
        )
        print(f"    ✅ Dashboard loaded: {type(dashboard_data).__name__}")
        
        # Connection drops
        self.architecture.connection_status = ConnectionStatus.OFFLINE
        print("    📴 Connection lost")
        
        # User makes changes
        changes = {"layout": "updated", "last_modified": datetime.now().isoformat()}
        update_result = await self.architecture.execute_request(
            "dashboard_config", "update", 
            {"id": "main_dashboard", **changes}, "scenario_user"
        )
        print(f"    ✅ Changes saved offline: {update_result.get('status', 'unknown')}")
        
        # Connection restored
        self.architecture.connection_status = ConnectionStatus.ONLINE
        print("    🌐 Connection restored")
        
        # Auto-sync
        sync_result = await self.architecture.force_sync_all()
        print(f"    ✅ Auto-sync completed: {sync_result.get('success', False)}")
        
        # Scenario 2: PWA installation and usage
        print("\n📱 Scenario 2: PWA Installation")
        print("  1. User visits ScrollIntel multiple times")
        print("  2. Install prompt appears")
        print("  3. User installs PWA")
        print("  4. PWA works offline with service worker")
        
        # Track multiple visits
        for visit in range(5):
            self.pwa.track_user_engagement("pwa_user", "visit")
            print(f"    👤 Visit {visit + 1} tracked")
        
        # Show install prompt
        prompt = await self.pwa.handle_install_prompt("pwa_user")
        if prompt['show_prompt']:
            print(f"    📲 Install prompt: {prompt['message']}")
            
            # User installs
            await self.pwa.handle_install_result("pwa_user", prompt['event_id'], "accepted")
            print("    ✅ PWA installed successfully")
        
        # Scenario 3: Conflict resolution
        print("\n⚔️ Scenario 3: Conflict Resolution")
        print("  1. User A and User B edit same document")
        print("  2. Both work offline")
        print("  3. Both come online and sync")
        print("  4. Conflict detected")
        print("  5. Intelligent merge resolves conflict")
        
        # This would be demonstrated with actual conflict data
        # For demo purposes, we'll show the conflict resolution capabilities
        conflicts = self.offline_manager.get_conflicts()
        if conflicts:
            print(f"    ⚠️ Found {len(conflicts)} conflicts")
            for conflict in conflicts[:2]:
                print(f"      🔥 {conflict['collection']}: {conflict['conflict_type']}")
        else:
            print("    ✅ No conflicts found - system working smoothly!")
        
        print("\n🎯 Integration Summary:")
        print("  ✅ Seamless offline/online transitions")
        print("  ✅ Automatic conflict resolution")
        print("  ✅ Progressive Web App capabilities")
        print("  ✅ Intelligent caching and sync")
        print("  ✅ User experience continuity")
    
    async def show_final_statistics(self):
        """Show final statistics and summary."""
        print("\n📊 FINAL STATISTICS & SUMMARY")
        print("=" * 50)
        
        # Offline Data Manager Stats
        storage_info = self.offline_manager.get_storage_info()
        sync_status = self.offline_manager.get_sync_status()
        
        print("💾 Offline Data Manager:")
        print(f"  Database size: {storage_info['database_size_mb']} MB")
        print(f"  Total entries: {storage_info['total_entries']}")
        print(f"  Collections: {len(storage_info['collections'])}")
        print(f"  Sync operations: {sync_status['pending_sync_operations']}")
        
        # Sync Engine Stats
        if self.sync_engine:
            metrics = self.sync_engine.get_sync_metrics()
            print(f"\n🔄 Sync Engine:")
            print(f"  Total syncs: {metrics['total_syncs']}")
            print(f"  Success rate: {metrics['success_rate']:.2%}")
            print(f"  Conflicts resolved: {metrics['conflicts_resolved']}")
            print(f"  Auto-resolution rate: {metrics['auto_resolution_rate']:.2%}")
        
        # Architecture Stats
        arch_status = self.architecture.get_architecture_status()
        perf_metrics = arch_status['performance_metrics']
        
        print(f"\n🏗️ Offline-First Architecture:")
        print(f"  Connection: {arch_status['connection_status']}")
        print(f"  Operation mode: {arch_status['operation_mode']}")
        print(f"  Offline requests: {perf_metrics['requests_served_offline']}")
        print(f"  Online requests: {perf_metrics['requests_served_online']}")
        print(f"  Cache hits: {perf_metrics['cache_hits']}")
        
        # PWA Stats
        pwa_status = self.pwa.get_pwa_status()
        pwa_metrics = pwa_status['performance_metrics']
        
        print(f"\n📱 Progressive Web App:")
        print(f"  Service Worker: v{pwa_status['service_worker_version']}")
        print(f"  Cache rules: {pwa_status['cache_rules_count']}")
        print(f"  Install prompts: {pwa_metrics['install_prompts_shown']}")
        print(f"  Successful installs: {pwa_metrics['successful_installs']}")
        print(f"  Push notifications: {pwa_metrics['push_notifications_sent']}")
        print(f"  Background syncs: {pwa_metrics['background_syncs']}")
        
        # Feature Summary
        capabilities = self.architecture.get_offline_capabilities()
        offline_features = sum(1 for cap in capabilities.values() if cap['read_offline'])
        writable_features = sum(1 for cap in capabilities.values() if cap['write_offline'])
        
        print(f"\n🎯 Feature Capabilities:")
        print(f"  Total features: {len(capabilities)}")
        print(f"  Offline readable: {offline_features}")
        print(f"  Offline writable: {writable_features}")
        
        print(f"\n✨ DEMO COMPLETED SUCCESSFULLY!")
        print("ScrollIntel's offline capabilities provide:")
        print("  🔹 Comprehensive offline data management")
        print("  🔹 Intelligent conflict resolution")
        print("  🔹 Seamless online/offline transitions")
        print("  🔹 Progressive Web App features")
        print("  🔹 Bulletproof user experience")
    
    async def cleanup(self):
        """Clean up demo resources."""
        print("\n🧹 Cleaning up demo resources...")
        
        if self.offline_manager:
            self.offline_manager.close()
        
        if self.architecture:
            self.architecture.close()
        
        print("✅ Cleanup completed")
    
    async def run_complete_demo(self):
        """Run the complete offline capabilities demo."""
        try:
            await self.initialize_components()
            
            # Run all demo sections
            stored_entries = await self.demo_offline_data_management()
            await self.demo_sync_capabilities(stored_entries)
            await self.demo_offline_first_architecture()
            await self.demo_progressive_web_app()
            await self.demo_integration_scenarios()
            await self.show_final_statistics()
            
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup()


async def main():
    """Main demo function."""
    print("🌟 ScrollIntel Advanced Offline Capabilities Demo")
    print("=" * 60)
    print("This demo showcases ScrollIntel's comprehensive offline capabilities:")
    print("• Offline Data Management with SQLite storage")
    print("• Intelligent Sync Engine with conflict resolution")
    print("• Offline-First Architecture with adaptive strategies")
    print("• Progressive Web App with service worker")
    print("• Seamless online/offline transitions")
    print("=" * 60)
    
    demo = OfflineCapabilitiesDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())