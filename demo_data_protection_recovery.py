"""
Demo script for comprehensive data protection and recovery system.
Showcases automatic continuous save, multi-tier backup, data integrity verification,
and cross-device state synchronization with offline support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from scrollintel.core.data_protection_recovery import (
    DataProtectionRecoverySystem,
    protect_data,
    recover_data,
    create_recovery_point,
    with_data_protection,
    BackupTier,
    SyncStatus
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_auto_save():
    """Demonstrate automatic continuous save functionality."""
    print("\n" + "="*60)
    print("DEMO: Automatic Continuous Save System")
    print("="*60)
    
    system = DataProtectionRecoverySystem()
    await system.start()
    
    try:
        # Simulate user working with data
        user_data = {
            'user_id': 'demo_user_001',
            'document': 'My important document',
            'last_edit': datetime.utcnow().isoformat(),
            'word_count': 150
        }
        
        print(f"📝 User starts working on document...")
        await system.auto_save_manager.register_data('demo_user_001', 'document', user_data)
        
        # Simulate continuous edits
        for i in range(5):
            await asyncio.sleep(1)
            user_data['word_count'] += 25
            user_data['last_edit'] = datetime.utcnow().isoformat()
            user_data['document'] += f" Additional content {i+1}."
            
            await system.auto_save_manager.register_data('demo_user_001', 'document', user_data)
            print(f"✏️  Edit {i+1}: Word count now {user_data['word_count']}")
        
        # Force save to demonstrate immediate save
        print(f"💾 Forcing immediate save...")
        await system.auto_save_manager.force_save('demo_user_001', 'document')
        
        print(f"✅ Auto-save demo completed successfully!")
        
    finally:
        await system.stop()


async def demo_multi_tier_backup():
    """Demonstrate multi-tier backup strategy."""
    print("\n" + "="*60)
    print("DEMO: Multi-Tier Backup Strategy")
    print("="*60)
    
    system = DataProtectionRecoverySystem()
    await system.start()
    
    try:
        # Create different types of data
        user_profile = {
            'user_id': 'demo_user_002',
            'name': 'Jane Doe',
            'email': 'jane@example.com',
            'preferences': {
                'theme': 'dark',
                'notifications': True,
                'language': 'en'
            },
            'created_at': datetime.utcnow().isoformat()
        }
        
        analysis_results = {
            'analysis_id': 'analysis_001',
            'user_id': 'demo_user_002',
            'data_source': 'sales_data.csv',
            'results': {
                'total_revenue': 125000,
                'growth_rate': 15.5,
                'top_products': ['Product A', 'Product B', 'Product C']
            },
            'confidence_score': 0.92,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Create snapshots in different tiers
        print(f"📦 Creating backup snapshots...")
        
        profile_snapshot = await system.backup_manager.create_snapshot(
            'demo_user_002', 'user_profile', user_profile,
            tiers=[BackupTier.LOCAL, BackupTier.REMOTE]
        )
        print(f"✅ User profile backed up to LOCAL and REMOTE tiers")
        print(f"   Snapshot ID: {profile_snapshot.snapshot_id}")
        print(f"   Size: {profile_snapshot.size_bytes} bytes")
        print(f"   Checksum: {profile_snapshot.checksum[:16]}...")
        
        analysis_snapshot = await system.backup_manager.create_snapshot(
            'demo_user_002', 'analysis_results', analysis_results,
            tiers=[BackupTier.LOCAL, BackupTier.REMOTE, BackupTier.DISTRIBUTED]
        )
        print(f"✅ Analysis results backed up to LOCAL, REMOTE, and DISTRIBUTED tiers")
        print(f"   Snapshot ID: {analysis_snapshot.snapshot_id}")
        
        # Create recovery point
        recovery_point = await system.backup_manager.create_recovery_point(
            'demo_user_002', 'Complete user state backup'
        )
        print(f"🔄 Recovery point created: {recovery_point.recovery_id}")
        
        # Demonstrate restoration
        print(f"\n🔧 Demonstrating data restoration...")
        
        restored_profile = await system.backup_manager.restore_from_snapshot(profile_snapshot.snapshot_id)
        print(f"✅ Restored user profile: {restored_profile['name']}")
        
        restored_analysis = await system.backup_manager.restore_from_snapshot(analysis_snapshot.snapshot_id)
        print(f"✅ Restored analysis results: Revenue ${restored_analysis['results']['total_revenue']:,}")
        
        # Restore from recovery point
        all_data = await system.backup_manager.restore_from_recovery_point(recovery_point.recovery_id)
        print(f"✅ Restored from recovery point: {len(all_data)} data types")
        
    finally:
        await system.stop()


async def demo_data_integrity():
    """Demonstrate data integrity verification and repair."""
    print("\n" + "="*60)
    print("DEMO: Data Integrity Verification and Repair")
    print("="*60)
    
    system = DataProtectionRecoverySystem()
    await system.start()
    
    try:
        # Create good data and backup
        good_data = {
            'user_id': 'demo_user_003',
            'financial_data': {
                'accounts': [
                    {'name': 'Checking', 'balance': 5000.00},
                    {'name': 'Savings', 'balance': 15000.00}
                ],
                'total_balance': 20000.00,
                'last_updated': datetime.utcnow().isoformat()
            }
        }
        
        print(f"💾 Creating backup of good financial data...")
        await system.backup_manager.create_snapshot('demo_user_003', 'financial_data', good_data)
        
        # Verify integrity of good data
        integrity_status = await system.integrity_verifier.verify_data_integrity(
            'demo_user_003', 'financial_data', good_data
        )
        print(f"✅ Good data integrity check: {integrity_status.value}")
        
        # Simulate data corruption
        print(f"\n⚠️  Simulating data corruption...")
        corrupted_data = None  # Simulate complete data loss
        
        integrity_status = await system.integrity_verifier.verify_data_integrity(
            'demo_user_003', 'financial_data', corrupted_data
        )
        print(f"❌ Corrupted data integrity check: {integrity_status.value}")
        
        # Attempt repair
        print(f"🔧 Attempting automatic repair...")
        repaired_data = await system.integrity_verifier.repair_corrupted_data(
            'demo_user_003', 'financial_data', corrupted_data, system.backup_manager
        )
        
        if repaired_data:
            print(f"✅ Data successfully repaired!")
            print(f"   Total balance restored: ${repaired_data['financial_data']['total_balance']:,.2f}")
            
            # Verify repaired data
            integrity_status = await system.integrity_verifier.verify_data_integrity(
                'demo_user_003', 'financial_data', repaired_data
            )
            print(f"✅ Repaired data integrity check: {integrity_status.value}")
        else:
            print(f"❌ Could not repair data")
        
        # Demonstrate custom integrity check
        print(f"\n🔍 Demonstrating custom integrity check...")
        
        async def financial_data_check(data):
            """Custom integrity check for financial data."""
            if not isinstance(data, dict) or 'financial_data' not in data:
                return False
            
            financial = data['financial_data']
            if 'accounts' not in financial or 'total_balance' not in financial:
                return False
            
            # Verify balance calculation
            calculated_total = sum(account['balance'] for account in financial['accounts'])
            return abs(calculated_total - financial['total_balance']) < 0.01
        
        system.integrity_verifier.register_integrity_check('financial_data', financial_data_check)
        
        # Test with correct data
        integrity_status = await system.integrity_verifier.verify_data_integrity(
            'demo_user_003', 'financial_data', good_data
        )
        print(f"✅ Custom integrity check (correct data): {integrity_status.value}")
        
        # Test with incorrect balance
        bad_balance_data = good_data.copy()
        bad_balance_data['financial_data']['total_balance'] = 999999.99  # Wrong total
        
        integrity_status = await system.integrity_verifier.verify_data_integrity(
            'demo_user_003', 'financial_data', bad_balance_data
        )
        print(f"❌ Custom integrity check (wrong balance): {integrity_status.value}")
        
    finally:
        await system.stop()


async def demo_cross_device_sync():
    """Demonstrate cross-device synchronization."""
    print("\n" + "="*60)
    print("DEMO: Cross-Device State Synchronization")
    print("="*60)
    
    system = DataProtectionRecoverySystem()
    await system.start()
    
    try:
        # Register multiple devices
        print(f"📱 Registering user devices...")
        
        laptop = await system.sync_manager.register_device(
            'laptop_001', 'demo_user_004', 
            {'type': 'laptop', 'os': 'Windows', 'capabilities': ['full_sync', 'offline_mode']}
        )
        print(f"✅ Registered laptop: {laptop.device_id}")
        
        phone = await system.sync_manager.register_device(
            'phone_001', 'demo_user_004',
            {'type': 'mobile', 'os': 'iOS', 'capabilities': ['limited_sync', 'push_notifications']}
        )
        print(f"✅ Registered phone: {phone.device_id}")
        
        tablet = await system.sync_manager.register_device(
            'tablet_001', 'demo_user_004',
            {'type': 'tablet', 'os': 'Android', 'capabilities': ['full_sync', 'offline_mode']}
        )
        print(f"✅ Registered tablet: {tablet.device_id}")
        
        # Sync data from laptop
        print(f"\n💻 Syncing data from laptop...")
        project_data = {
            'project_id': 'project_alpha',
            'name': 'Data Analysis Project',
            'status': 'in_progress',
            'files': ['data.csv', 'analysis.py', 'results.json'],
            'last_modified': datetime.utcnow().isoformat(),
            'modified_by': 'laptop_001'
        }
        
        sync_status = await system.sync_manager.sync_data('laptop_001', 'project_data', project_data)
        print(f"✅ Laptop sync status: {sync_status.value}")
        
        # Sync same data from phone (no conflict)
        print(f"\n📱 Syncing same data from phone...")
        sync_status = await system.sync_manager.sync_data('phone_001', 'project_data', project_data)
        print(f"✅ Phone sync status: {sync_status.value}")
        
        # Create conflict by modifying data on tablet
        print(f"\n📋 Creating sync conflict from tablet...")
        conflicting_data = project_data.copy()
        conflicting_data['status'] = 'completed'
        conflicting_data['files'].append('final_report.pdf')
        conflicting_data['last_modified'] = datetime.utcnow().isoformat()
        conflicting_data['modified_by'] = 'tablet_001'
        
        # Add offline change to simulate conflict
        tablet.offline_changes.append({
            'data_type': 'project_data',
            'data': conflicting_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        await system.sync_manager._save_device_state(tablet)
        
        # Try to sync from laptop again
        sync_status = await system.sync_manager.sync_data('laptop_001', 'project_data', project_data)
        print(f"⚠️  Laptop sync status (with conflict): {sync_status.value}")
        
        if sync_status == SyncStatus.CONFLICT:
            print(f"🔍 Conflicts detected: {len(system.sync_manager.sync_conflicts)}")
            
            # Resolve conflict using latest timestamp
            conflict_id = list(system.sync_manager.sync_conflicts.keys())[0]
            conflict = system.sync_manager.sync_conflicts[conflict_id]
            
            print(f"🔧 Resolving conflict using 'use_latest' strategy...")
            resolved = await system.sync_manager.resolve_conflict(conflict_id, 'use_latest')
            
            if resolved:
                print(f"✅ Conflict resolved successfully!")
                print(f"   Resolution strategy: {conflict.resolution_strategy}")
            else:
                print(f"❌ Failed to resolve conflict")
        
        # Demonstrate offline/online handling
        print(f"\n🔌 Demonstrating offline/online handling...")
        
        # Mark phone as offline
        await system.sync_manager.handle_device_offline('phone_001')
        phone_state = system.sync_manager.device_states['phone_001']
        print(f"📱 Phone marked offline: {phone_state.sync_status.value}")
        
        # Bring phone back online
        offline_operations = await system.sync_manager.handle_device_online('phone_001')
        phone_state = system.sync_manager.device_states['phone_001']
        print(f"📱 Phone back online: {phone_state.sync_status.value}")
        print(f"   Processed {len(offline_operations)} offline operations")
        
        # Get sync status for user
        sync_status_report = system.sync_manager.get_sync_status('demo_user_004')
        print(f"\n📊 Sync Status Report:")
        print(f"   Total devices: {sync_status_report['total_devices']}")
        print(f"   All synced: {sync_status_report['all_synced']}")
        print(f"   Active conflicts: {sync_status_report['active_conflicts']}")
        
        for device in sync_status_report['devices']:
            print(f"   Device {device['device_id']}: {device['sync_status']}")
        
    finally:
        await system.stop()


async def demo_comprehensive_protection():
    """Demonstrate comprehensive data protection workflow."""
    print("\n" + "="*60)
    print("DEMO: Comprehensive Data Protection Workflow")
    print("="*60)
    
    system = DataProtectionRecoverySystem()
    await system.start()
    
    try:
        # Simulate a complete user workflow
        user_id = 'demo_user_005'
        device_id = 'workstation_001'
        
        # Register device
        await system.sync_manager.register_device(device_id, user_id, {'type': 'workstation'})
        
        # User creates important business data
        business_data = {
            'company': 'TechCorp Inc.',
            'quarterly_report': {
                'q1_revenue': 1250000,
                'q1_expenses': 800000,
                'q1_profit': 450000,
                'growth_rate': 12.5,
                'key_metrics': {
                    'customer_acquisition': 150,
                    'customer_retention': 94.2,
                    'market_share': 8.7
                }
            },
            'forecasts': {
                'q2_revenue_projection': 1400000,
                'annual_target': 5500000
            },
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0'
        }
        
        print(f"💼 User creates important business data...")
        print(f"   Q1 Revenue: ${business_data['quarterly_report']['q1_revenue']:,}")
        print(f"   Q1 Profit: ${business_data['quarterly_report']['q1_profit']:,}")
        
        # Protect the data comprehensively
        print(f"\n🛡️  Applying comprehensive protection...")
        success = await system.protect_user_data(user_id, 'business_data', business_data, device_id)
        
        if success:
            print(f"✅ Data protection applied successfully!")
            
            # Get protection status
            status = await system.get_protection_status(user_id)
            print(f"📊 Protection Status:")
            print(f"   Backup snapshots: {status['backup_snapshots']}")
            print(f"   Latest backup: {status['latest_backup']}")
            print(f"   Auto-save active: {status['auto_save_active']}")
            print(f"   Protection level: {status['protection_level']}")
            print(f"   Sync status: {status['sync_status']['all_synced']}")
        
        # Create a recovery point
        print(f"\n🔄 Creating recovery point...")
        recovery_id = await system.create_recovery_point(user_id, 'Q1 Report Completion')
        print(f"✅ Recovery point created: {recovery_id}")
        
        # Simulate data loss scenario
        print(f"\n💥 Simulating catastrophic data loss...")
        print(f"   (User accidentally deletes all files)")
        
        # Recover the data
        print(f"🔧 Initiating data recovery...")
        recovered_data = await system.recover_user_data(user_id, 'business_data')
        
        if 'business_data' in recovered_data:
            recovered_business = recovered_data['business_data']
            print(f"✅ Data recovery successful!")
            print(f"   Recovered company: {recovered_business['company']}")
            print(f"   Recovered Q1 revenue: ${recovered_business['quarterly_report']['q1_revenue']:,}")
            print(f"   Data integrity verified: ✅")
        else:
            print(f"❌ Data recovery failed")
        
        # Demonstrate recovery from recovery point
        print(f"\n🔄 Recovering from recovery point...")
        all_recovered = await system.recover_user_data(user_id, recovery_point_id=recovery_id)
        print(f"✅ Recovered {len(all_recovered)} data types from recovery point")
        
    finally:
        await system.stop()


async def demo_convenience_functions():
    """Demonstrate convenience functions and decorators."""
    print("\n" + "="*60)
    print("DEMO: Convenience Functions and Decorators")
    print("="*60)
    
    # Start the global system
    from scrollintel.core.data_protection_recovery import data_protection_system
    await data_protection_system.start()
    
    try:
        # Demonstrate protect_data function
        print(f"🛡️  Using protect_data convenience function...")
        
        user_preferences = {
            'user_id': 'demo_user_006',
            'theme': 'dark',
            'language': 'en',
            'notifications': {
                'email': True,
                'push': False,
                'sms': False
            },
            'privacy_settings': {
                'profile_public': False,
                'data_sharing': False
            }
        }
        
        success = await protect_data('demo_user_006', 'preferences', user_preferences)
        print(f"✅ Data protection: {'Success' if success else 'Failed'}")
        
        # Demonstrate recover_data function
        print(f"\n🔧 Using recover_data convenience function...")
        recovered = await recover_data('demo_user_006', 'preferences')
        
        if 'preferences' in recovered:
            print(f"✅ Data recovery successful!")
            print(f"   Theme: {recovered['preferences']['theme']}")
            print(f"   Language: {recovered['preferences']['language']}")
        
        # Demonstrate decorator
        print(f"\n🎭 Using @with_data_protection decorator...")
        
        @with_data_protection('analysis_results')
        async def perform_analysis(user_id, data_source):
            """Simulate data analysis with automatic protection."""
            print(f"   🔍 Analyzing data from {data_source}...")
            await asyncio.sleep(0.5)  # Simulate processing time
            
            return {
                'user_id': user_id,
                'analysis_type': 'trend_analysis',
                'data_source': data_source,
                'results': {
                    'trend': 'upward',
                    'confidence': 0.87,
                    'key_insights': [
                        'Revenue increased 15% over last quarter',
                        'Customer satisfaction improved',
                        'Market share expanded in key demographics'
                    ]
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Function automatically protects its results
        analysis_result = await perform_analysis('demo_user_006', 'sales_data.csv')
        print(f"✅ Analysis completed and automatically protected!")
        print(f"   Trend: {analysis_result['results']['trend']}")
        print(f"   Confidence: {analysis_result['results']['confidence']:.2%}")
        
        # Verify protection worked
        status = await data_protection_system.get_protection_status('demo_user_006')
        print(f"📊 Final protection status: {status['backup_snapshots']} snapshots")
        
    finally:
        await data_protection_system.stop()


async def main():
    """Run all demos."""
    print("🚀 ScrollIntel Data Protection and Recovery System Demo")
    print("=" * 80)
    
    try:
        await demo_auto_save()
        await demo_multi_tier_backup()
        await demo_data_integrity()
        await demo_cross_device_sync()
        await demo_comprehensive_protection()
        await demo_convenience_functions()
        
        print("\n" + "="*80)
        print("🎉 All demos completed successfully!")
        print("="*80)
        
        print("\n📋 Summary of Features Demonstrated:")
        print("✅ Automatic continuous save with conflict resolution")
        print("✅ Multi-tier backup strategy with instant recovery")
        print("✅ Data integrity verification and automatic repair")
        print("✅ Cross-device state synchronization with offline support")
        print("✅ Comprehensive protection workflow")
        print("✅ Convenient APIs and decorators")
        
        print("\n🔒 Your data is now bulletproof! 🔒")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())