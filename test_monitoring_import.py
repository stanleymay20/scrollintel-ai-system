"""Test script to debug monitoring import issues."""

try:
    print("1. Importing analytics...")
    from scrollintel.core.bulletproof_monitoring_analytics import bulletproof_analytics
    print("   ✅ Analytics imported successfully")
    
    print("2. Importing monitoring module...")
    import scrollintel.core.bulletproof_monitoring as monitoring_module
    print("   ✅ Monitoring module imported successfully")
    
    print("3. Checking module attributes...")
    attrs = dir(monitoring_module)
    print(f"   Module attributes: {[attr for attr in attrs if not attr.startswith('_')]}")
    
    print("4. Trying to access BulletproofMonitoring class...")
    if hasattr(monitoring_module, 'BulletproofMonitoring'):
        print("   ✅ BulletproofMonitoring class found")
        cls = getattr(monitoring_module, 'BulletproofMonitoring')
        print(f"   Class: {cls}")
    else:
        print("   ❌ BulletproofMonitoring class not found")
    
    print("5. Trying to access bulletproof_monitoring instance...")
    if hasattr(monitoring_module, 'bulletproof_monitoring'):
        print("   ✅ bulletproof_monitoring instance found")
        instance = getattr(monitoring_module, 'bulletproof_monitoring')
        print(f"   Instance: {instance}")
    else:
        print("   ❌ bulletproof_monitoring instance not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()