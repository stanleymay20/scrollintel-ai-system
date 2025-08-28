#!/usr/bin/env python3
"""
ScrollIntel Continuous Monitoring Dashboard
Real-time monitoring of ScrollIntel deployment status
"""

import time
import os
import sys
from datetime import datetime
import json
from check_deployment_status import ScrollIntelDeploymentChecker

class ContinuousMonitor:
    def __init__(self, interval=30):
        self.interval = interval
        self.checker = ScrollIntelDeploymentChecker()
        self.history = []
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def display_header(self):
        """Display monitoring header"""
        print("🔄 ScrollIntel Continuous Monitoring Dashboard")
        print(f"⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 Refresh Interval: {self.interval}s")
        print("=" * 80)
        
    def display_quick_status(self, results):
        """Display quick status overview"""
        healthy = sum(1 for r in results if "✅" in r.status)
        total = len(results)
        
        status_emoji = "🟢" if healthy == total else "🟡" if healthy > total * 0.7 else "🔴"
        
        print(f"\n{status_emoji} System Status: {healthy}/{total} services healthy ({healthy/total*100:.0f}%)")
        
        # Show critical services only
        critical_services = [r for r in results if any(keyword in r.name.lower() 
                           for keyword in ['backend', 'frontend', 'database', 'health'])]
        
        print("\n🔧 Critical Services:")
        for service in critical_services[:5]:  # Show top 5
            status_icon = "✅" if "✅" in service.status else "❌" if "❌" in service.status else "⚠️"
            print(f"  {status_icon} {service.name}")
            
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        try:
            # Run quick check (subset of full check)
            core_services = [
                ("http://localhost:8000/health", "Backend Health"),
                ("http://localhost:3000", "Frontend App"),
            ]
            
            results = []
            for url, name in core_services:
                result = self.checker.check_service(url, name)
                results.append(result)
                
            # Add database check
            db_result = self.checker.check_database_connection()
            results.append(db_result)
            
            # Store in history
            self.history.append({
                "timestamp": datetime.now(),
                "healthy_count": sum(1 for r in results if "✅" in r.status),
                "total_count": len(results)
            })
            
            # Keep only last 20 entries
            if len(self.history) > 20:
                self.history = self.history[-20:]
                
            return results
            
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            return []
            
    def display_history_trend(self):
        """Display simple trend information"""
        if len(self.history) < 2:
            return
            
        print("\n📈 Recent Trend (last 10 checks):")
        recent = self.history[-10:]
        
        trend_line = ""
        for entry in recent:
            ratio = entry["healthy_count"] / entry["total_count"]
            if ratio == 1.0:
                trend_line += "🟢"
            elif ratio > 0.7:
                trend_line += "🟡"
            else:
                trend_line += "🔴"
                
        print(f"  {trend_line}")
        print(f"  (🟢=Healthy 🟡=Degraded 🔴=Critical)")
        
    def run(self):
        """Run continuous monitoring"""
        print("🚀 Starting ScrollIntel Continuous Monitoring...")
        print(f"⏰ Checking every {self.interval} seconds")
        print("💡 Press Ctrl+C to stop")
        print("\n" + "=" * 50)
        
        try:
            while True:
                self.clear_screen()
                self.display_header()
                
                results = self.run_monitoring_cycle()
                
                if results:
                    self.display_quick_status(results)
                    self.display_history_trend()
                else:
                    print("❌ Failed to get monitoring results")
                
                print(f"\n⏳ Next check in {self.interval} seconds...")
                print("💡 Press Ctrl+C to stop monitoring")
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print("📊 Final Status Summary:")
            if self.history:
                last_check = self.history[-1]
                print(f"  Last Check: {last_check['healthy_count']}/{last_check['total_count']} services healthy")
            print("👋 Goodbye!")
            
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
            sys.exit(1)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ScrollIntel Continuous Monitoring")
    parser.add_argument("--interval", "-i", type=int, default=30, 
                       help="Check interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    monitor = ContinuousMonitor(interval=args.interval)
    monitor.run()

if __name__ == "__main__":
    main()