#!/usr/bin/env python3
"""
ScrollIntel Launch Coordinator
Manages the step-by-step launch process across all phases
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class LaunchCoordinator:
    """Coordinates the ScrollIntel launch process"""
    
    def __init__(self):
        self.launch_config = self.load_launch_config()
        self.current_phase = self.get_current_phase()
        self.launch_log = []
        
    def load_launch_config(self) -> Dict:
        """Load launch configuration"""
        config_file = "launch_config.json"
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = {
                "launch_start_date": datetime.now().isoformat(),
                "phases": {
                    "1": {
                        "name": "Local Development Validation",
                        "duration_days": 1,
                        "status": "pending",
                        "script": "scripts/phase1-local-validation-windows.py"
                    },
                    "2": {
                        "name": "Staging Environment Setup",
                        "duration_days": 2,
                        "status": "pending",
                        "script": "scripts/phase2-staging-setup.py"
                    },
                    "3": {
                        "name": "Limited Production Beta",
                        "duration_days": 4,
                        "status": "pending",
                        "script": "scripts/phase3-production-beta.py"
                    },
                    "4": {
                        "name": "Soft Launch",
                        "duration_days": 7,
                        "status": "pending",
                        "script": "scripts/phase4-soft-launch.py"
                    },
                    "5": {
                        "name": "Full Production Launch",
                        "duration_days": 0,
                        "status": "pending",
                        "script": "scripts/phase5-full-launch.py"
                    }
                },
                "environments": {
                    "local": {
                        "url": "http://localhost:8000",
                        "status": "not_configured"
                    },
                    "staging": {
                        "url": "https://staging.scrollintel.com",
                        "status": "not_configured"
                    },
                    "production": {
                        "url": "https://api.scrollintel.com",
                        "status": "not_configured"
                    }
                },
                "monitoring": {
                    "prometheus_url": "http://localhost:9090",
                    "grafana_url": "http://localhost:3000",
                    "status": "not_configured"
                },
                "notifications": {
                    "slack_webhook": "",
                    "email_alerts": [],
                    "status": "not_configured"
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def save_launch_config(self):
        """Save current launch configuration"""
        with open("launch_config.json", 'w') as f:
            json.dump(self.launch_config, f, indent=2)
    
    def get_current_phase(self) -> int:
        """Determine current launch phase"""
        for phase_num, phase_info in self.launch_config["phases"].items():
            if phase_info["status"] in ["pending", "in_progress"]:
                return int(phase_num)
        return 1  # Default to phase 1
    
    def log_event(self, event_type: str, message: str, phase: Optional[int] = None):
        """Log launch event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase or self.current_phase,
            "type": event_type,
            "message": message
        }
        
        self.launch_log.append(log_entry)
        
        # Also print to console
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {event_type.upper()}: {message}")
    
    def print_launch_status(self):
        """Print current launch status"""
        print("🚀 ScrollIntel Launch Status")
        print("=" * 50)
        print(f"Current Phase: {self.current_phase}")
        print(f"Launch Started: {self.launch_config['launch_start_date']}")
        print()
        
        for phase_num, phase_info in self.launch_config["phases"].items():
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(phase_info["status"], "❓")
            
            current_marker = " <- CURRENT" if int(phase_num) == self.current_phase else ""
            
            print(f"{status_icon} Phase {phase_num}: {phase_info['name']}{current_marker}")
        
        print()
    
    def run_phase(self, phase_num: int) -> bool:
        """Run a specific launch phase"""
        phase_info = self.launch_config["phases"][str(phase_num)]
        
        self.log_event("phase_start", f"Starting Phase {phase_num}: {phase_info['name']}", phase_num)
        
        # Update phase status
        phase_info["status"] = "in_progress"
        phase_info["start_time"] = datetime.now().isoformat()
        self.save_launch_config()
        
        # Run phase script
        script_path = phase_info.get("script")
        if script_path and os.path.exists(script_path):
            try:
                self.log_event("script_start", f"Running {script_path}", phase_num)
                
                result = subprocess.run([
                    sys.executable, script_path
                ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                
                if result.returncode == 0:
                    self.log_event("script_success", f"Phase {phase_num} script completed successfully", phase_num)
                    phase_info["status"] = "completed"
                    phase_info["end_time"] = datetime.now().isoformat()
                    self.save_launch_config()
                    return True
                else:
                    self.log_event("script_error", f"Phase {phase_num} script failed: {result.stderr}", phase_num)
                    phase_info["status"] = "failed"
                    phase_info["error"] = result.stderr
                    phase_info["end_time"] = datetime.now().isoformat()
                    self.save_launch_config()
                    return False
                    
            except subprocess.TimeoutExpired:
                self.log_event("script_timeout", f"Phase {phase_num} script timed out", phase_num)
                phase_info["status"] = "failed"
                phase_info["error"] = "Script execution timed out"
                phase_info["end_time"] = datetime.now().isoformat()
                self.save_launch_config()
                return False
            except Exception as e:
                self.log_event("script_exception", f"Phase {phase_num} script error: {e}", phase_num)
                phase_info["status"] = "failed"
                phase_info["error"] = str(e)
                phase_info["end_time"] = datetime.now().isoformat()
                self.save_launch_config()
                return False
        else:
            self.log_event("script_missing", f"Phase {phase_num} script not found: {script_path}", phase_num)
            # For now, mark as completed if no script (manual phase)
            phase_info["status"] = "completed"
            phase_info["end_time"] = datetime.now().isoformat()
            self.save_launch_config()
            return True
    
    def run_next_phase(self) -> bool:
        """Run the next pending phase"""
        return self.run_phase(self.current_phase)
    
    def run_all_phases(self):
        """Run all phases sequentially"""
        self.log_event("launch_start", "Starting ScrollIntel launch process")
        
        for phase_num in range(1, 6):  # Phases 1-5
            if not self.run_phase(phase_num):
                self.log_event("launch_failed", f"Launch failed at Phase {phase_num}")
                return False
            
            # Update current phase
            self.current_phase = phase_num + 1
        
        self.log_event("launch_complete", "ScrollIntel launch completed successfully!")
        return True
    
    def rollback_phase(self, phase_num: int):
        """Rollback a specific phase"""
        phase_info = self.launch_config["phases"][str(phase_num)]
        
        self.log_event("rollback_start", f"Rolling back Phase {phase_num}", phase_num)
        
        # Reset phase status
        phase_info["status"] = "pending"
        if "start_time" in phase_info:
            del phase_info["start_time"]
        if "end_time" in phase_info:
            del phase_info["end_time"]
        if "error" in phase_info:
            del phase_info["error"]
        
        self.save_launch_config()
        self.log_event("rollback_complete", f"Phase {phase_num} rolled back", phase_num)
    
    def get_launch_summary(self) -> Dict:
        """Get launch summary statistics"""
        total_phases = len(self.launch_config["phases"])
        completed_phases = sum(1 for p in self.launch_config["phases"].values() 
                             if p["status"] == "completed")
        failed_phases = sum(1 for p in self.launch_config["phases"].values() 
                          if p["status"] == "failed")
        
        return {
            "total_phases": total_phases,
            "completed_phases": completed_phases,
            "failed_phases": failed_phases,
            "success_rate": (completed_phases / total_phases) * 100,
            "current_phase": self.current_phase,
            "launch_events": len(self.launch_log)
        }
    
    def print_launch_summary(self):
        """Print launch summary"""
        summary = self.get_launch_summary()
        
        print("\n📊 Launch Summary")
        print("=" * 30)
        print(f"Total Phases: {summary['total_phases']}")
        print(f"Completed: {summary['completed_phases']}")
        print(f"Failed: {summary['failed_phases']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Current Phase: {summary['current_phase']}")
        print(f"Total Events: {summary['launch_events']}")
    
    def interactive_menu(self):
        """Interactive launch coordination menu"""
        while True:
            print("\n🚀 ScrollIntel Launch Coordinator")
            print("=" * 40)
            print("1. Show Launch Status")
            print("2. Run Next Phase")
            print("3. Run Specific Phase")
            print("4. Run All Phases")
            print("5. Rollback Phase")
            print("6. Show Launch Summary")
            print("7. View Launch Log")
            print("8. Exit")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == "1":
                self.print_launch_status()
            
            elif choice == "2":
                print(f"\nRunning Phase {self.current_phase}...")
                success = self.run_next_phase()
                if success:
                    print(f"✅ Phase {self.current_phase - 1} completed successfully!")
                else:
                    print(f"❌ Phase {self.current_phase} failed!")
            
            elif choice == "3":
                phase_num = input("Enter phase number (1-5): ").strip()
                try:
                    phase_num = int(phase_num)
                    if 1 <= phase_num <= 5:
                        print(f"\nRunning Phase {phase_num}...")
                        success = self.run_phase(phase_num)
                        if success:
                            print(f"✅ Phase {phase_num} completed successfully!")
                        else:
                            print(f"❌ Phase {phase_num} failed!")
                    else:
                        print("❌ Invalid phase number. Must be 1-5.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
            
            elif choice == "4":
                print("\nRunning all phases...")
                confirm = input("This will run all phases sequentially. Continue? (y/N): ")
                if confirm.lower() == 'y':
                    success = self.run_all_phases()
                    if success:
                        print("🎉 All phases completed successfully!")
                    else:
                        print("❌ Launch process failed!")
                else:
                    print("Launch cancelled.")
            
            elif choice == "5":
                phase_num = input("Enter phase number to rollback (1-5): ").strip()
                try:
                    phase_num = int(phase_num)
                    if 1 <= phase_num <= 5:
                        confirm = input(f"Rollback Phase {phase_num}? (y/N): ")
                        if confirm.lower() == 'y':
                            self.rollback_phase(phase_num)
                            print(f"✅ Phase {phase_num} rolled back!")
                        else:
                            print("Rollback cancelled.")
                    else:
                        print("❌ Invalid phase number. Must be 1-5.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
            
            elif choice == "6":
                self.print_launch_summary()
            
            elif choice == "7":
                print("\n📋 Launch Log (last 10 events):")
                for event in self.launch_log[-10:]:
                    timestamp = event["timestamp"][:19]  # Remove microseconds
                    print(f"[{timestamp}] Phase {event['phase']} - {event['type']}: {event['message']}")
            
            elif choice == "8":
                print("Goodbye! 👋")
                break
            
            else:
                print("❌ Invalid option. Please select 1-8.")

def main():
    """Main function"""
    coordinator = LaunchCoordinator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            coordinator.print_launch_status()
        elif command == "next":
            success = coordinator.run_next_phase()
            sys.exit(0 if success else 1)
        elif command == "all":
            success = coordinator.run_all_phases()
            sys.exit(0 if success else 1)
        elif command == "summary":
            coordinator.print_launch_summary()
        elif command.startswith("phase"):
            try:
                phase_num = int(command.replace("phase", ""))
                success = coordinator.run_phase(phase_num)
                sys.exit(0 if success else 1)
            except ValueError:
                print("❌ Invalid phase number")
                sys.exit(1)
        else:
            print("❌ Unknown command. Use: status, next, all, summary, or phase<N>")
            sys.exit(1)
    else:
        # Interactive mode
        coordinator.interactive_menu()

if __name__ == "__main__":
    main()