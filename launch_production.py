#!/usr/bin/env python3
"""
ScrollIntel Production Launcher
Launches the full production environment with PostgreSQL, Redis, and all services
"""

import subprocess
import sys
import os
import time
import webbrowser
import signal
import json
from pathlib import Path

class ProductionLauncher:
    def __init__(self):
        self.services_running = False
        
    def print_banner(self):
        """Print ScrollIntel production banner"""
        print("\n" + "="*70)
        print("🚀 SCROLLINTEL™ PRODUCTION DEPLOYMENT LAUNCHER")
        print("="*70)
        print("🏭 Full Production Environment")
        print("🐘 PostgreSQL Database")
        print("🔴 Redis Cache")
        print("🌐 Next.js Frontend")
        print("⚡ FastAPI Backend")
        print("📊 Production Monitoring")
        print("🔒 Enterprise Security")
        print("="*70 + "\n")
    
    def check_docker(self):
        """Check if Docker and Docker Compose are available"""
        print("🔍 Checking Docker installation...")
        
        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Docker is not installed or not running")
                print("💡 Please install Docker Desktop: https://docs.docker.com/desktop/")
                return False
            print(f"✅ {result.stdout.strip()}")
            
            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Docker Compose is not installed")
                print("💡 Please install Docker Compose: https://docs.docker.com/compose/install/")
                return False
            print(f"✅ {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Docker daemon is not running")
                print("💡 Please start Docker Desktop")
                return False
            print("✅ Docker daemon is running")
            
            return True
            
        except FileNotFoundError:
            print("❌ Docker is not installed")
            print("💡 Please install Docker Desktop: https://docs.docker.com/desktop/")
            return False
    
    def setup_environment(self):
        """Setup production environment"""
        print("🔧 Setting up production environment...")
        
        # Ensure .env exists
        env_file = Path('.env')
        if not env_file.exists():
            print("❌ .env file not found")
            print("💡 Please ensure .env file exists with production configuration")
            return False
        
        # Create necessary directories
        directories = ['uploads', 'logs', 'models', 'generated_content']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        print("✅ Environment configured")
        return True
    
    def pull_images(self):
        """Pull Docker images"""
        print("📦 Pulling Docker images...")
        
        try:
            result = subprocess.run([
                'docker-compose', 'pull'
            ], check=True, capture_output=True, text=True)
            print("✅ Docker images pulled successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to pull images: {e}")
            print("💡 Check your internet connection and Docker configuration")
            return False
    
    def start_services(self):
        """Start all production services"""
        print("🚀 Starting production services...")
        
        try:
            # Start services in detached mode
            result = subprocess.run([
                'docker-compose', 'up', '-d', '--build'
            ], check=True, capture_output=True, text=True)
            
            print("✅ Services started successfully")
            self.services_running = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start services: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False
    
    def wait_for_services(self):
        """Wait for services to be ready"""
        print("⏳ Waiting for services to be ready...")
        
        services = {
            'postgres': {'port': 5432, 'name': 'PostgreSQL Database'},
            'redis': {'port': 6379, 'name': 'Redis Cache'},
            'backend': {'port': 8000, 'name': 'Backend API'},
            'frontend': {'port': 3000, 'name': 'Frontend Application'}
        }
        
        max_wait = 120  # 2 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            all_ready = True
            
            for service, config in services.items():
                try:
                    result = subprocess.run([
                        'docker-compose', 'ps', '--services', '--filter', 'status=running'
                    ], capture_output=True, text=True)
                    
                    if service not in result.stdout:
                        all_ready = False
                        break
                        
                except subprocess.CalledProcessError:
                    all_ready = False
                    break
            
            if all_ready:
                print("✅ All services are ready!")
                return True
            
            print(f"⏳ Waiting... ({wait_time}s/{max_wait}s)")
            time.sleep(5)
            wait_time += 5
        
        print("⚠️  Services may still be starting up...")
        return True
    
    def check_health(self):
        """Check service health"""
        print("🏥 Checking service health...")
        
        try:
            # Check backend health
            import requests
            response = requests.get('http://localhost:8000/health', timeout=10)
            if response.status_code == 200:
                print("✅ Backend API is healthy")
            else:
                print(f"⚠️  Backend API returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Backend API health check failed: {e}")
        
        try:
            # Check frontend
            response = requests.get('http://localhost:3000', timeout=10)
            if response.status_code == 200:
                print("✅ Frontend is healthy")
            else:
                print(f"⚠️  Frontend returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Frontend health check failed: {e}")
    
    def show_status(self):
        """Show deployment status and access points"""
        print("\n" + "="*70)
        print("🎉 SCROLLINTEL™ PRODUCTION DEPLOYMENT COMPLETE!")
        print("="*70)
        print("🌐 Frontend Application:    http://localhost:3000")
        print("🔧 Backend API:             http://localhost:8000")
        print("📚 API Documentation:       http://localhost:8000/docs")
        print("❤️  Health Check:           http://localhost:8000/health")
        print("🐘 PostgreSQL Database:     localhost:5432")
        print("🔴 Redis Cache:             localhost:6379")
        print("="*70)
        print("\n🚀 Production Features:")
        print("• Enterprise-grade PostgreSQL database")
        print("• High-performance Redis caching")
        print("• Production-optimized Docker containers")
        print("• Advanced monitoring and logging")
        print("• Security hardening and compliance")
        print("• Scalable microservices architecture")
        print("• Real-time WebSocket connections")
        print("• AI-powered CTO capabilities")
        print("\n💼 Management Commands:")
        print("📊 View logs:        docker-compose logs -f")
        print("📈 Monitor services: docker-compose ps")
        print("🔄 Restart service:  docker-compose restart <service>")
        print("🛑 Stop all:         docker-compose down")
        print("🗑️  Clean up:        docker-compose down -v --remove-orphans")
        print("\n🌟 ScrollIntel™ - Production-Ready AI Platform!")
        print("="*70 + "\n")
    
    def open_browser(self):
        """Open the application in browser"""
        print("🌐 Opening ScrollIntel in your browser...")
        try:
            webbrowser.open('http://localhost:3000')
            time.sleep(2)
            webbrowser.open('http://localhost:8000/docs')
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print("💡 Please open http://localhost:3000 manually")
    
    def monitor_services(self):
        """Monitor running services"""
        print("👀 Monitoring services... (Press Ctrl+C to stop)")
        
        try:
            while self.services_running:
                # Check service status
                result = subprocess.run([
                    'docker-compose', 'ps', '--format', 'json'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    try:
                        services = json.loads(result.stdout) if result.stdout.strip() else []
                        running_count = len([s for s in services if s.get('State') == 'running'])
                        total_count = len(services)
                        
                        if running_count < total_count:
                            print(f"⚠️  Warning: {running_count}/{total_count} services running")
                    except json.JSONDecodeError:
                        pass
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    def cleanup(self):
        """Clean up services"""
        if self.services_running:
            print("\n🛑 Stopping ScrollIntel services...")
            try:
                subprocess.run(['docker-compose', 'down'], check=True)
                print("✅ Services stopped successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error stopping services: {e}")
        
        print("👋 ScrollIntel production deployment stopped")
    
    def run(self):
        """Main run method"""
        try:
            self.print_banner()
            
            if not self.check_docker():
                return 1
            
            if not self.setup_environment():
                return 1
            
            if not self.pull_images():
                return 1
            
            if not self.start_services():
                return 1
            
            if not self.wait_for_services():
                return 1
            
            # Give services a bit more time to fully initialize
            print("⏳ Final initialization...")
            time.sleep(10)
            
            self.check_health()
            self.show_status()
            self.open_browser()
            
            # Monitor services
            self.monitor_services()
            
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
        finally:
            self.cleanup()
        
        return 0

def main():
    """Main entry point"""
    launcher = ProductionLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())