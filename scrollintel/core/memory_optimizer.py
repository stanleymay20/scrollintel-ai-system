"""
Memory Optimization Module
"""
import gc
import psutil
import threading
import time
from typing import Optional

class MemoryOptimizer:
    """Memory optimization and monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.cleanup_threshold = 85  # Memory usage percentage
        
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self.monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.cleanup_threshold:
                    self.cleanup_memory()
                time.sleep(30)  # Check every 30 seconds
            except Exception:
                pass
                
    def cleanup_memory(self):
        """Perform memory cleanup"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear caches if available
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(700, 10, 10)
                
            return collected
        except Exception:
            return 0
            
    def get_memory_info(self):
        """Get current memory information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'system_memory_percent': system_memory.percent,
                'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
            }
        except Exception:
            return {}

# Global memory optimizer
memory_optimizer = MemoryOptimizer()

def optimize_memory():
    """Quick memory optimization function"""
    return memory_optimizer.cleanup_memory()

def start_memory_monitoring():
    """Start automatic memory monitoring"""
    memory_optimizer.start_monitoring()
