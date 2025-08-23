#!/usr/bin/env python3
"""
Check which ports are available for ScrollIntel
"""

import socket
import subprocess
import sys

def check_port(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False

def get_port_process(port):
    """Get the process using a specific port (Windows)"""
    try:
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True, 
            shell=True
        )
        
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        # Get process name
                        tasklist_result = subprocess.run(
                            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
                            capture_output=True,
                            text=True,
                            shell=True
                        )
                        lines = tasklist_result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            process_name = lines[1].split(',')[0].strip('"')
                            return f"{process_name} (PID: {pid})"
                    except:
                        return f"PID: {pid}"
        return "Unknown process"
    except:
        return "Could not determine"

def main():
    print("🔍 ScrollIntel Port Availability Check")
    print("=" * 40)
    
    ports_to_check = [8000, 8001, 8002, 8003, 8080, 3000, 3001]
    available_ports = []
    
    for port in ports_to_check:
        if check_port(port):
            print(f"✅ Port {port}: Available")
            available_ports.append(port)
        else:
            process = get_port_process(port)
            print(f"❌ Port {port}: In use by {process}")
    
    print("\n" + "=" * 40)
    if available_ports:
        print(f"🎉 Available ports: {', '.join(map(str, available_ports))}")
        print(f"💡 ScrollIntel will try to use port {available_ports[0]}")
    else:
        print("⚠️  No common ports available!")
        print("💡 Try closing other applications or restart your computer")
    
    print("\n🔧 To free up port 8000, you can:")
    print("   1. Close any running development servers")
    print("   2. Check for other Python/Node.js processes")
    print("   3. Restart your computer if needed")

if __name__ == "__main__":
    main()
    input("\nPress Enter to continue...")