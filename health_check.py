#!/usr/bin/env python3
import requests
import sys
from datetime import datetime

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ScrollIntel is healthy")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to ScrollIntel (not running?)")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🏥 ScrollIntel Health Check")
    print("=" * 30)
    
    if check_health():
        print("\n🎉 ScrollIntel is running properly!")
        sys.exit(0)
    else:
        print("\n💡 Try starting ScrollIntel with:")
        print("   python run_simple.py")
        print("   or")
        print("   ./start_scrollintel.sh")
        sys.exit(1)
