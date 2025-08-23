#!/usr/bin/env python3
"""
ScrollIntel Performance Benchmark
Tests application performance and optimization
"""

import time
import psutil
import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Any

class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "benchmarks": {},
            "recommendations": []
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('.').percent,
            "python_version": sys.version
        }
    
    def benchmark_import_speed(self) -> Dict[str, Any]:
        """Benchmark module import speed"""
        print("🚀 Benchmarking import speed...")
        
        imports_to_test = [
            "scrollintel.core.config",
            "scrollintel.api.main",
            "scrollintel.core.monitoring",
            "scrollintel.agents.base",
            "scrollintel.models.database"
        ]
        
        results = {}
        total_time = 0
        
        for module in imports_to_test:
            start_time = time.time()
            try:
                __import__(module)
                import_time = time.time() - start_time
                results[module] = {
                    "time": round(import_time, 4),
                    "status": "success"
                }
                total_time += import_time
                print(f"  ✅ {module}: {import_time:.4f}s")
            except Exception as e:
                results[module] = {
                    "time": 0,
                    "status": "failed",
                    "error": str(e)
                }
                print(f"  ❌ {module}: FAILED - {e}")
        
        return {
            "individual_imports": results,
            "total_time": round(total_time, 4),
            "average_time": round(total_time / len(imports_to_test), 4),
            "score": 100 if total_time < 2.0 else max(0, 100 - (total_time * 10))
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage"""
        print("\n💾 Benchmarking memory usage...")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Import heavy modules
        try:
            import tensorflow as tf
            import numpy as np
            import pandas as pd
            heavy_imports_memory = process.memory_info().rss / 1024 / 1024
        except:
            heavy_imports_memory = initial_memory
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        
        results = {
            "initial_memory_mb": round(initial_memory, 2),
            "after_imports_mb": round(heavy_imports_memory, 2),
            "memory_increase_mb": round(heavy_imports_memory - initial_memory, 2),
            "system_memory_percent": system_memory.percent,
            "available_memory_gb": round(system_memory.available / 1024 / 1024 / 1024, 2)
        }
        
        # Calculate score
        if system_memory.percent < 70:
            score = 100
        elif system_memory.percent < 85:
            score = 80
        else:
            score = 50
            
        results["score"] = score
        
        print(f"  📊 Initial memory: {results['initial_memory_mb']} MB")
        print(f"  📊 After imports: {results['after_imports_mb']} MB")
        print(f"  📊 System memory usage: {results['system_memory_percent']}%")
        
        return results
    
    def benchmark_cpu_performance(self) -> Dict[str, Any]:
        """Benchmark CPU performance"""
        print("\n🔥 Benchmarking CPU performance...")
        
        # CPU intensive task
        start_time = time.time()
        result = sum(i * i for i in range(100000))
        cpu_task_time = time.time() - start_time
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        results = {
            "cpu_task_time": round(cpu_task_time, 4),
            "cpu_usage_percent": cpu_percent,
            "cpu_count": psutil.cpu_count(),
            "computation_result": result
        }
        
        # Calculate score
        if cpu_task_time < 0.01:
            score = 100
        elif cpu_task_time < 0.05:
            score = 80
        else:
            score = max(0, 100 - (cpu_task_time * 1000))
            
        results["score"] = score
        
        print(f"  ⚡ CPU task time: {cpu_task_time:.4f}s")
        print(f"  📊 CPU usage: {cpu_percent}%")
        
        return results
    
    async def benchmark_async_performance(self) -> Dict[str, Any]:
        """Benchmark async performance"""
        print("\n⚡ Benchmarking async performance...")
        
        # Test async operations
        start_time = time.time()
        
        async def async_task(n):
            await asyncio.sleep(0.001)  # Simulate async work
            return n * n
        
        # Run concurrent tasks
        tasks = [async_task(i) for i in range(100)]
        results_list = await asyncio.gather(*tasks)
        
        async_time = time.time() - start_time
        
        results = {
            "async_task_time": round(async_time, 4),
            "tasks_completed": len(results_list),
            "tasks_per_second": round(len(results_list) / async_time, 2)
        }
        
        # Calculate score
        if async_time < 0.2:
            score = 100
        elif async_time < 0.5:
            score = 80
        else:
            score = max(0, 100 - (async_time * 100))
            
        results["score"] = score
        
        print(f"  ⚡ Async tasks time: {async_time:.4f}s")
        print(f"  📊 Tasks per second: {results['tasks_per_second']}")
        
        return results
    
    def benchmark_file_io(self) -> Dict[str, Any]:
        """Benchmark file I/O performance"""
        print("\n📁 Benchmarking file I/O performance...")
        
        # Write test
        test_data = "x" * 10000  # 10KB of data
        start_time = time.time()
        
        with open("test_file.txt", "w") as f:
            for i in range(100):
                f.write(test_data)
        
        write_time = time.time() - start_time
        
        # Read test
        start_time = time.time()
        
        with open("test_file.txt", "r") as f:
            content = f.read()
        
        read_time = time.time() - start_time
        
        # Cleanup
        import os
        os.remove("test_file.txt")
        
        results = {
            "write_time": round(write_time, 4),
            "read_time": round(read_time, 4),
            "total_io_time": round(write_time + read_time, 4),
            "data_size_mb": round(len(content) / 1024 / 1024, 2)
        }
        
        # Calculate score
        total_time = write_time + read_time
        if total_time < 0.1:
            score = 100
        elif total_time < 0.5:
            score = 80
        else:
            score = max(0, 100 - (total_time * 100))
            
        results["score"] = score
        
        print(f"  📝 Write time: {write_time:.4f}s")
        print(f"  📖 Read time: {read_time:.4f}s")
        
        return results
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("🏁 Starting ScrollIntel Performance Benchmark")
        print("=" * 60)
        
        # Run benchmarks
        self.results["benchmarks"] = {
            "import_speed": self.benchmark_import_speed(),
            "memory_usage": self.benchmark_memory_usage(),
            "cpu_performance": self.benchmark_cpu_performance(),
            "async_performance": await self.benchmark_async_performance(),
            "file_io": self.benchmark_file_io()
        }
        
        # Calculate overall score
        scores = [bench["score"] for bench in self.results["benchmarks"].values()]
        overall_score = sum(scores) / len(scores)
        
        self.results["overall_score"] = round(overall_score, 2)
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.results
    
    def generate_recommendations(self):
        """Generate performance recommendations"""
        benchmarks = self.results["benchmarks"]
        recommendations = []
        
        if benchmarks["import_speed"]["score"] < 70:
            recommendations.append("🚀 Optimize module imports - consider lazy loading")
        
        if benchmarks["memory_usage"]["score"] < 70:
            recommendations.append("💾 Optimize memory usage - implement memory pooling")
        
        if benchmarks["cpu_performance"]["score"] < 70:
            recommendations.append("🔥 Optimize CPU usage - consider multiprocessing")
        
        if benchmarks["async_performance"]["score"] < 70:
            recommendations.append("⚡ Optimize async operations - review concurrency patterns")
        
        if benchmarks["file_io"]["score"] < 70:
            recommendations.append("📁 Optimize file I/O - implement caching and buffering")
        
        if self.results["overall_score"] >= 90:
            recommendations.append("✨ Excellent performance! System is well optimized")
        elif self.results["overall_score"] >= 75:
            recommendations.append("👍 Good performance with room for improvement")
        else:
            recommendations.append("⚠️ Performance needs significant optimization")
        
        self.results["recommendations"] = recommendations
    
    def print_results(self):
        """Print formatted benchmark results"""
        results = self.results
        
        print("\n" + "=" * 60)
        print("📊 SCROLLINTEL PERFORMANCE BENCHMARK REPORT")
        print("=" * 60)
        
        # Overall score
        score = results["overall_score"]
        if score >= 90:
            emoji = "🟢"
            status = "EXCELLENT"
        elif score >= 75:
            emoji = "🟡"
            status = "GOOD"
        else:
            emoji = "🔴"
            status = "NEEDS OPTIMIZATION"
        
        print(f"\n{emoji} OVERALL PERFORMANCE SCORE: {score}/100 ({status})")
        
        # Individual benchmark scores
        print("\n📋 BENCHMARK BREAKDOWN:")
        for name, data in results["benchmarks"].items():
            score = data["score"]
            emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"  {emoji} {name.replace('_', ' ').title()}: {score}/100")
        
        # System info
        print(f"\n🖥️ SYSTEM INFO:")
        sys_info = results["system_info"]
        print(f"  • CPU Cores: {sys_info['cpu_count']}")
        print(f"  • Memory: {sys_info['memory_total'] // (1024**3)} GB")
        print(f"  • Disk Usage: {sys_info['disk_usage']}%")
        
        # Recommendations
        if results["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                print(f"  • {rec}")
        
        print("\n" + "=" * 60)
        print(f"📅 Benchmark completed at: {results['timestamp']}")
        print("=" * 60)

async def main():
    """Main function"""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    benchmark.print_results()
    
    # Save results
    with open('performance_benchmark_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: performance_benchmark_report.json")
    
    # Return exit code based on score
    return 0 if results["overall_score"] >= 75 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))