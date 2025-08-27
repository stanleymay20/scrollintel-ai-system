"""
Production load testing for visual generation system.
Tests system behavior under realistic production conditions.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from scrollintel.engines.visual_generation.engine import VisualGenerationEngine
from scrollintel.engines.visual_generation.pipeline import ImageGenerationPipeline
from scrollintel.engines.visual_generation.models.ultra_performance_pipeline import UltraRealisticVideoGenerationPipeline
from scrollintel.core.monitoring import MetricsCollector
from scrollintel.core.user_management import UserManager


class ProductionLoadTester:
    """Production-grade load testing for visual generation system."""
    
    def __init__(self):
        self.visual_engine = VisualGenerationEngine()
        self.image_pipeline = ImageGenerationPipeline()
        self.video_pipeline = UltraRealisticVideoGenerationPipeline()
        self.metrics_collector = MetricsCollector()
        self.user_manager = UserManager()
        
        # Load test configuration
        self.load_test_config = {
            "concurrent_users": 50,
            "requests_per_user": 10,
            "ramp_up_time": 30,  # seconds
            "test_duration": 300,  # 5 minutes
            "max_response_time": 30,  # seconds
            "min_success_rate": 0.95
        }
        
    async def test_production_image_load(self):
        """Test image generation under production load."""
        
        print("🚀 Starting production image load test...")
        
        # Create test users
        test_users = await self._create_test_users(self.load_test_config["concurrent_users"])
        
        # Prepare test requests
        test_prompts = [
            "Professional business headshot, high quality",
            "Modern office environment, photorealistic",
            "Corporate team meeting, professional lighting",
            "Executive portrait, studio quality",
            "Business presentation scene, clean background"
        ]
        
        # Start load test
        start_time = time.time()
        results = []
        
        # Ramp up users gradually
        user_batches = self._create_user_batches(test_users, self.load_test_config["ramp_up_time"])
        
        for batch in user_batches:
            batch_tasks = []
            
            for user in batch:
                for i in range(self.load_test_config["requests_per_user"]):
                    prompt = test_prompts[i % len(test_prompts)]
                    task = asyncio.create_task(
                        self._execute_image_request(user["id"], prompt, i)
                    )
                    batch_tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Small delay between batches for ramp-up
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_load_test_results(results, total_time)
        
        # Validate performance requirements
        assert analysis["success_rate"] >= self.load_test_config["min_success_rate"]
        assert analysis["avg_response_time"] <= self.load_test_config["max_response_time"]
        assert analysis["p95_response_time"] <= self.load_test_config["max_response_time"] * 1.5
        
        print(f"✅ Production image load test passed:")
        print(f"   - Total requests: {analysis['total_requests']}")
        print(f"   - Success rate: {analysis['success_rate']:.2%}")
        print(f"   - Average response time: {analysis['avg_response_time']:.2f}s")
        print(f"   - P95 response time: {analysis['p95_response_time']:.2f}s")
        print(f"   - Throughput: {analysis['throughput']:.2f} req/s")
        
        return analysis
        
    async def test_production_video_load(self):
        """Test ultra-realistic video generation under production load."""
        
        print("🎬 Starting production video load test...")
        
        # Smaller concurrent load for video due to resource intensity
        video_config = {
            "concurrent_users": 10,
            "requests_per_user": 3,
            "max_response_time": 60
        }
        
        test_users = await self._create_test_users(video_config["concurrent_users"])
        
        video_prompts = [
            "Professional businesswoman presenting, ultra-realistic, 4K",
            "Executive giving speech, photorealistic quality",
            "Team collaboration meeting, broadcast quality"
        ]
        
        start_time = time.time()
        video_tasks = []
        
        for user in test_users:
            for i in range(video_config["requests_per_user"]):
                prompt = video_prompts[i % len(video_prompts)]
                task = asyncio.create_task(
                    self._execute_video_request(user["id"], prompt, i)
                )
                video_tasks.append(task)
        
        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent video generation
        
        async def limited_video_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_video_task(task) for task in video_tasks]
        video_results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze video results
        video_analysis = self._analyze_video_load_results(video_results, total_time)
        
        # Validate video performance
        assert video_analysis["success_rate"] >= 0.90  # Slightly lower for video
        assert video_analysis["avg_response_time"] <= video_config["max_response_time"]
        assert video_analysis["avg_quality_score"] >= 0.95  # Ultra-realistic requirement
        
        print(f"✅ Production video load test passed:")
        print(f"   - Total video requests: {video_analysis['total_requests']}")
        print(f"   - Success rate: {video_analysis['success_rate']:.2%}")
        print(f"   - Average response time: {video_analysis['avg_response_time']:.2f}s")
        print(f"   - Average quality score: {video_analysis['avg_quality_score']:.3f}")
        print(f"   - Average humanoid accuracy: {video_analysis['avg_humanoid_accuracy']:.3f}")
        
        return video_analysis
        
    async def test_mixed_workload_stress(self):
        """Test system under mixed image and video workload."""
        
        print("⚡ Starting mixed workload stress test...")
        
        # Create mixed workload
        mixed_users = await self._create_test_users(30)
        
        # 70% image requests, 30% video requests
        image_users = mixed_users[:21]
        video_users = mixed_users[21:]
        
        # Start both workloads simultaneously
        image_task = asyncio.create_task(
            self._run_image_workload(image_users, requests_per_user=5)
        )
        
        video_task = asyncio.create_task(
            self._run_video_workload(video_users, requests_per_user=2)
        )
        
        # Monitor system resources during test
        resource_monitor_task = asyncio.create_task(
            self._monitor_system_resources(duration=180)
        )
        
        # Wait for all tasks to complete
        image_results, video_results, resource_metrics = await asyncio.gather(
            image_task, video_task, resource_monitor_task
        )
        
        # Validate mixed workload performance
        assert image_results["success_rate"] >= 0.90
        assert video_results["success_rate"] >= 0.85
        assert resource_metrics["max_cpu_usage"] <= 90.0
        assert resource_metrics["max_memory_usage"] <= 85.0
        
        print(f"✅ Mixed workload stress test passed:")
        print(f"   - Image success rate: {image_results['success_rate']:.2%}")
        print(f"   - Video success rate: {video_results['success_rate']:.2%}")
        print(f"   - Max CPU usage: {resource_metrics['max_cpu_usage']:.1f}%")
        print(f"   - Max memory usage: {resource_metrics['max_memory_usage']:.1f}%")
        
        return {
            "image_results": image_results,
            "video_results": video_results,
            "resource_metrics": resource_metrics
        }
        
    async def test_sustained_load_endurance(self):
        """Test system endurance under sustained load."""
        
        print("🏃 Starting sustained load endurance test...")
        
        endurance_config = {
            "duration": 600,  # 10 minutes
            "concurrent_users": 20,
            "request_interval": 5,  # seconds between requests
        }
        
        test_users = await self._create_test_users(endurance_config["concurrent_users"])
        
        start_time = time.time()
        endurance_results = []
        
        # Run sustained load for specified duration
        while time.time() - start_time < endurance_config["duration"]:
            batch_tasks = []
            
            for user in test_users:
                task = asyncio.create_task(
                    self._execute_image_request(
                        user["id"], 
                        "Sustained load test image", 
                        int(time.time() - start_time)
                    )
                )
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            endurance_results.extend(batch_results)
            
            # Wait before next batch
            await asyncio.sleep(endurance_config["request_interval"])
        
        total_duration = time.time() - start_time
        
        # Analyze endurance results
        endurance_analysis = self._analyze_endurance_results(endurance_results, total_duration)
        
        # Validate endurance performance
        assert endurance_analysis["success_rate"] >= 0.95
        assert endurance_analysis["performance_degradation"] <= 0.10  # Max 10% degradation
        assert endurance_analysis["memory_leak_detected"] == False
        
        print(f"✅ Sustained load endurance test passed:")
        print(f"   - Test duration: {total_duration:.1f}s")
        print(f"   - Total requests: {endurance_analysis['total_requests']}")
        print(f"   - Success rate: {endurance_analysis['success_rate']:.2%}")
        print(f"   - Performance degradation: {endurance_analysis['performance_degradation']:.2%}")
        
        return endurance_analysis
        
    async def _create_test_users(self, count: int) -> List[Dict[str, Any]]:
        """Create test users for load testing."""
        users = []
        
        for i in range(count):
            user_data = {
                "email": f"loadtest_{i}@scrollintel.com",
                "password": f"LoadTest{i}123!",
                "username": f"loaduser_{i}"
            }
            
            try:
                user_response = await self.user_manager.create_user(**user_data)
                users.append({
                    "id": user_response.user_id,
                    "email": user_data["email"]
                })
            except Exception as e:
                print(f"Warning: Could not create user {i}: {e}")
                
        return users
        
    def _create_user_batches(self, users: List[Dict], ramp_up_time: int) -> List[List[Dict]]:
        """Create user batches for gradual ramp-up."""
        batch_size = max(1, len(users) // ramp_up_time)
        batches = []
        
        for i in range(0, len(users), batch_size):
            batch = users[i:i + batch_size]
            batches.append(batch)
            
        return batches
        
    async def _execute_image_request(self, user_id: str, prompt: str, request_id: int) -> Dict[str, Any]:
        """Execute single image generation request."""
        start_time = time.time()
        
        try:
            request = {
                "prompt": prompt,
                "resolution": (1024, 1024),
                "quality": "high",
                "num_images": 1
            }
            
            result = await self.image_pipeline.generate_image(request, user_id)
            
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "response_time": response_time,
                "quality_score": result.quality_metrics.overall_score,
                "user_id": user_id,
                "request_id": request_id
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e),
                "user_id": user_id,
                "request_id": request_id
            }
            
    async def _execute_video_request(self, user_id: str, prompt: str, request_id: int) -> Dict[str, Any]:
        """Execute single video generation request."""
        start_time = time.time()
        
        try:
            request = {
                "prompt": prompt,
                "duration": 3.0,
                "resolution": (1920, 1080),
                "fps": 60,
                "style": "ultra_realistic"
            }
            
            result = await self.video_pipeline.generate_ultra_realistic_video(request, user_id)
            
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "response_time": response_time,
                "quality_score": result.quality_score,
                "humanoid_accuracy": result.humanoid_accuracy or 0.0,
                "user_id": user_id,
                "request_id": request_id
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e),
                "user_id": user_id,
                "request_id": request_id
            }
            
    def _analyze_load_test_results(self, results: List[Any], total_time: float) -> Dict[str, Any]:
        """Analyze load test results and calculate metrics."""
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        successful_results = [r for r in valid_results if r.get("success", False)]
        
        if not valid_results:
            return {"success_rate": 0.0, "avg_response_time": 0.0}
        
        success_rate = len(successful_results) / len(valid_results)
        
        response_times = [r["response_time"] for r in successful_results]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else avg_response_time
        
        throughput = len(valid_results) / total_time if total_time > 0 else 0.0
        
        return {
            "total_requests": len(valid_results),
            "successful_requests": len(successful_results),
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "throughput": throughput
        }
        
    def _analyze_video_load_results(self, results: List[Any], total_time: float) -> Dict[str, Any]:
        """Analyze video load test results."""
        
        valid_results = [r for r in results if isinstance(r, dict)]
        successful_results = [r for r in valid_results if r.get("success", False)]
        
        if not valid_results:
            return {"success_rate": 0.0}
        
        success_rate = len(successful_results) / len(valid_results)
        
        response_times = [r["response_time"] for r in successful_results]
        quality_scores = [r["quality_score"] for r in successful_results if "quality_score" in r]
        humanoid_accuracies = [r["humanoid_accuracy"] for r in successful_results if "humanoid_accuracy" in r]
        
        return {
            "total_requests": len(valid_results),
            "success_rate": success_rate,
            "avg_response_time": statistics.mean(response_times) if response_times else 0.0,
            "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0.0,
            "avg_humanoid_accuracy": statistics.mean(humanoid_accuracies) if humanoid_accuracies else 0.0
        }
        
    async def _run_image_workload(self, users: List[Dict], requests_per_user: int) -> Dict[str, Any]:
        """Run image generation workload."""
        tasks = []
        
        for user in users:
            for i in range(requests_per_user):
                task = asyncio.create_task(
                    self._execute_image_request(user["id"], f"Mixed workload image {i}", i)
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._analyze_load_test_results(results, 0)
        
    async def _run_video_workload(self, users: List[Dict], requests_per_user: int) -> Dict[str, Any]:
        """Run video generation workload."""
        tasks = []
        
        for user in users:
            for i in range(requests_per_user):
                task = asyncio.create_task(
                    self._execute_video_request(user["id"], f"Mixed workload video {i}", i)
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._analyze_video_load_results(results, 0)
        
    async def _monitor_system_resources(self, duration: int) -> Dict[str, Any]:
        """Monitor system resources during load test."""
        
        cpu_readings = []
        memory_readings = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            cpu_readings.append(cpu_percent)
            memory_readings.append(memory_percent)
            
            await asyncio.sleep(5)
        
        return {
            "max_cpu_usage": max(cpu_readings) if cpu_readings else 0.0,
            "avg_cpu_usage": statistics.mean(cpu_readings) if cpu_readings else 0.0,
            "max_memory_usage": max(memory_readings) if memory_readings else 0.0,
            "avg_memory_usage": statistics.mean(memory_readings) if memory_readings else 0.0
        }
        
    def _analyze_endurance_results(self, results: List[Any], total_duration: float) -> Dict[str, Any]:
        """Analyze endurance test results."""
        
        valid_results = [r for r in results if isinstance(r, dict)]
        successful_results = [r for r in valid_results if r.get("success", False)]
        
        if not valid_results:
            return {"success_rate": 0.0, "performance_degradation": 1.0}
        
        success_rate = len(successful_results) / len(valid_results)
        
        # Calculate performance degradation over time
        response_times = [r["response_time"] for r in successful_results]
        
        if len(response_times) >= 10:
            first_half = response_times[:len(response_times)//2]
            second_half = response_times[len(response_times)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            performance_degradation = (second_avg - first_avg) / first_avg if first_avg > 0 else 0.0
        else:
            performance_degradation = 0.0
        
        # Simple memory leak detection
        gc.collect()
        memory_leak_detected = psutil.virtual_memory().percent > 90.0
        
        return {
            "total_requests": len(valid_results),
            "success_rate": success_rate,
            "performance_degradation": max(0.0, performance_degradation),
            "memory_leak_detected": memory_leak_detected
        }


@pytest.mark.asyncio
async def test_production_load_comprehensive():
    """Run comprehensive production load testing."""
    
    load_tester = ProductionLoadTester()
    
    print("🚀 Starting comprehensive production load testing...")
    
    # Run all load tests
    image_results = await load_tester.test_production_image_load()
    video_results = await load_tester.test_production_video_load()
    mixed_results = await load_tester.test_mixed_workload_stress()
    endurance_results = await load_tester.test_sustained_load_endurance()
    
    # Generate summary report
    print("\n📊 PRODUCTION LOAD TEST SUMMARY:")
    print("=" * 50)
    print(f"Image Generation Load Test: {'✅ PASSED' if image_results['success_rate'] >= 0.95 else '❌ FAILED'}")
    print(f"Video Generation Load Test: {'✅ PASSED' if video_results['success_rate'] >= 0.90 else '❌ FAILED'}")
    print(f"Mixed Workload Stress Test: {'✅ PASSED' if mixed_results['image_results']['success_rate'] >= 0.90 else '❌ FAILED'}")
    print(f"Sustained Load Endurance Test: {'✅ PASSED' if endurance_results['success_rate'] >= 0.95 else '❌ FAILED'}")
    
    print("\n🎯 Performance Metrics:")
    print(f"- Image throughput: {image_results['throughput']:.2f} req/s")
    print(f"- Video avg response time: {video_results['avg_response_time']:.2f}s")
    print(f"- System max CPU usage: {mixed_results['resource_metrics']['max_cpu_usage']:.1f}%")
    print(f"- Endurance performance degradation: {endurance_results['performance_degradation']:.2%}")
    
    return {
        "image_results": image_results,
        "video_results": video_results,
        "mixed_results": mixed_results,
        "endurance_results": endurance_results
    }


if __name__ == "__main__":
    asyncio.run(test_production_load_comprehensive())