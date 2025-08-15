#!/usr/bin/env python3
"""
ScrollIntel Monitoring System Startup Script
Starts all monitoring components for comprehensive system monitoring
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
import subprocess
import time
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scrollintel.core.monitoring import performance_monitor, metrics_collector
from scrollintel.core.uptime_monitor import uptime_monitor
from scrollintel.core.log_aggregation import log_aggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/monitoring.log')
    ]
)

logger = logging.getLogger(__name__)

class MonitoringOrchestrator:
    """Orchestrates all monitoring components"""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        self.processes = []
        
    async def start_prometheus(self):
        """Start Prometheus server"""
        try:
            logger.info("Starting Prometheus...")
            
            # Check if Prometheus is already running
            try:
                result = subprocess.run(['pgrep', '-f', 'prometheus'], capture_output=True, text=True)
                if result.stdout.strip():
                    logger.info("Prometheus is already running")
                    return
            except:
                pass
            
            # Start Prometheus
            prometheus_cmd = [
                'prometheus',
                '--config.file=monitoring/prometheus.yml',
                '--storage.tsdb.path=data/prometheus',
                '--web.console.libraries=/usr/share/prometheus/console_libraries',
                '--web.console.templates=/usr/share/prometheus/consoles',
                '--web.enable-lifecycle',
                '--web.listen-address=0.0.0.0:9090'
            ]
            
            process = subprocess.Popen(
                prometheus_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=project_root
            )
            
            self.processes.append(('prometheus', process))
            logger.info("Prometheus started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus: {e}")
    
    async def start_grafana(self):
        """Start Grafana server"""
        try:
            logger.info("Starting Grafana...")
            
            # Check if Grafana is already running
            try:
                result = subprocess.run(['pgrep', '-f', 'grafana'], capture_output=True, text=True)
                if result.stdout.strip():
                    logger.info("Grafana is already running")
                    return
            except:
                pass
            
            # Start Grafana
            grafana_cmd = [
                'grafana-server',
                '--config=/etc/grafana/grafana.ini',
                '--homepath=/usr/share/grafana'
            ]
            
            process = subprocess.Popen(
                grafana_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes.append(('grafana', process))
            logger.info("Grafana started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Grafana: {e}")
    
    async def start_alertmanager(self):
        """Start Alertmanager"""
        try:
            logger.info("Starting Alertmanager...")
            
            # Check if Alertmanager is already running
            try:
                result = subprocess.run(['pgrep', '-f', 'alertmanager'], capture_output=True, text=True)
                if result.stdout.strip():
                    logger.info("Alertmanager is already running")
                    return
            except:
                pass
            
            # Start Alertmanager
            alertmanager_cmd = [
                'alertmanager',
                '--config.file=monitoring/alertmanager.yml',
                '--storage.path=data/alertmanager',
                '--web.listen-address=0.0.0.0:9093'
            ]
            
            process = subprocess.Popen(
                alertmanager_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=project_root
            )
            
            self.processes.append(('alertmanager', process))
            logger.info("Alertmanager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Alertmanager: {e}")
    
    async def setup_grafana_dashboards(self):
        """Setup Grafana dashboards"""
        try:
            logger.info("Setting up Grafana dashboards...")
            
            # Wait for Grafana to be ready
            await asyncio.sleep(10)
            
            # Import dashboard (simplified - would use Grafana API)
            dashboard_file = project_root / "monitoring" / "grafana-dashboard-comprehensive.json"
            if dashboard_file.exists():
                logger.info("Dashboard configuration found")
                # In a real implementation, you would use Grafana API to import
                # For now, just log that the dashboard is available
                logger.info("Dashboard ready for manual import at http://localhost:3000")
            
        except Exception as e:
            logger.error(f"Failed to setup Grafana dashboards: {e}")
    
    async def start_monitoring_components(self):
        """Start all monitoring components"""
        try:
            logger.info("Starting monitoring components...")
            
            # Setup structured logging
            log_aggregator.setup_logging()
            
            # Start performance monitoring
            performance_task = asyncio.create_task(performance_monitor.monitor_loop())
            self.tasks.append(('performance_monitor', performance_task))
            
            # Start uptime monitoring
            uptime_task = asyncio.create_task(uptime_monitor.start_monitoring())
            self.tasks.append(('uptime_monitor', uptime_task))
            
            # Start log aggregation background tasks
            log_flush_task = asyncio.create_task(self._periodic_log_flush())
            self.tasks.append(('log_aggregator', log_flush_task))
            
            logger.info("All monitoring components started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring components: {e}")
    
    async def _periodic_log_flush(self):
        """Periodically flush log buffer"""
        while self.running:
            try:
                await log_aggregator.flush_buffer()
                await asyncio.sleep(60)  # Flush every minute
            except Exception as e:
                logger.error(f"Error in log flush: {e}")
                await asyncio.sleep(60)
    
    async def health_check_loop(self):
        """Periodic health check of monitoring components"""
        while self.running:
            try:
                # Check if processes are still running
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"{name} process has stopped")
                        # Could restart the process here
                
                # Check if tasks are still running
                for name, task in self.tasks:
                    if task.done():
                        logger.error(f"{name} task has completed unexpectedly")
                        if task.exception():
                            logger.error(f"{name} task exception: {task.exception()}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)
    
    async def start(self):
        """Start all monitoring services"""
        logger.info("Starting ScrollIntel Monitoring System...")
        self.running = True
        
        try:
            # Create necessary directories
            os.makedirs('data/prometheus', exist_ok=True)
            os.makedirs('data/alertmanager', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            
            # Start external monitoring tools
            await self.start_prometheus()
            await self.start_alertmanager()
            await self.start_grafana()
            
            # Setup Grafana dashboards
            await self.setup_grafana_dashboards()
            
            # Start internal monitoring components
            await self.start_monitoring_components()
            
            # Start health check loop
            health_task = asyncio.create_task(self.health_check_loop())
            self.tasks.append(('health_check', health_task))
            
            logger.info("ScrollIntel Monitoring System started successfully")
            logger.info("Access points:")
            logger.info("  - Prometheus: http://localhost:9090")
            logger.info("  - Grafana: http://localhost:3000 (admin/admin)")
            logger.info("  - Alertmanager: http://localhost:9093")
            logger.info("  - Status Page: http://localhost:8000/status")
            
            # Wait for all tasks
            await asyncio.gather(*[task for _, task in self.tasks])
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop all monitoring services"""
        logger.info("Stopping ScrollIntel Monitoring System...")
        self.running = False
        
        # Cancel all tasks
        for name, task in self.tasks:
            if not task.done():
                logger.info(f"Cancelling {name} task...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop all processes
        for name, process in self.processes:
            if process.poll() is None:
                logger.info(f"Stopping {name} process...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} process...")
                    process.kill()
        
        logger.info("ScrollIntel Monitoring System stopped")

def signal_handler(orchestrator):
    """Handle shutdown signals"""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(orchestrator.stop())
    return handler

async def main():
    """Main entry point"""
    orchestrator = MonitoringOrchestrator()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler(orchestrator))
    signal.signal(signal.SIGTERM, signal_handler(orchestrator))
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    # Check if required tools are installed
    required_tools = ['prometheus', 'grafana-server', 'alertmanager']
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run(['which', tool], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error(f"Missing required tools: {', '.join(missing_tools)}")
        logger.error("Please install the missing tools before running the monitoring system")
        logger.info("Installation instructions:")
        logger.info("  - Prometheus: https://prometheus.io/download/")
        logger.info("  - Grafana: https://grafana.com/grafana/download")
        logger.info("  - Alertmanager: https://prometheus.io/download/#alertmanager")
        sys.exit(1)
    
    # Run the monitoring system
    asyncio.run(main())