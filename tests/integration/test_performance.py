"""
Performance Tests for ScrollIntel
Tests system performance under various load conditions
"""
import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from unittest.mock import patch, Mock
import statistics

from scrollintel.core.registry import AgentRegistry
from scrollintel.agents.scroll_cto_agent import ScrollCTOAgent
from scrollintel.agents.scroll_data_scientist import ScrollDataScientist
from scrollintel.agents.scroll_ml_engineer import ScrollMLEngineer


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'error_count': 0,
            'success_count': 0
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            self.metrics['cpu_usage'].append(psutil.cpu_percent())
            self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
            time.sleep(0.1)
    
    def record_response_time(self, response_time: float):
        """Record response time"""
        self.metrics['response_times'].append(response_time)
    
    def record_success(self):
        """Record successful operation"""
        self.metrics['success_count'] += 1
    
    def record_error(self):
        """Record error"""
        self.metrics['error_count'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'avg_cpu_usage': statistics.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'max_cpu_usage': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'avg_memory_usage': statistics.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'max_memory_usage': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'avg_response_time': statistics.mean(self.metrics['response_times']) if self.metrics['response_times'] else 0,
            'max_response_time': max(self.metrics['response_times']) if self.metrics['response_times'] else 0,
            'min_response_time': min(self.metrics['response_times']) if self.metrics['response_times'] else 0,
            'total_requests': self.metrics['success_count'] + self.metrics['error_count'],
            'success_rate': self.metrics['success_count'] / (self.metrics['success_count'] + self.metrics['error_count']) if (self.metrics['success_count'] + self.metrics['error_count']) > 0 else 0,
            'error_rate': self.metrics['error_count'] / (self.metrics['success_count'] + self.metrics['error_count']) if (self.metrics['success_count'] + self.metrics['error_count']) > 0 else 0
        }


class TestPerformance:
    """Performance test suite"""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_requests_performance(self, agent_registry, mock_ai_services):
        """Test performance with concurrent agent requests"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup agents
        agents = [
            ScrollCTOAgent(),
            ScrollDataScientist(),
            ScrollMLEngineer()
        ]
        
        for agent in agents:
            agent_registry.register_agent(agent)
        
        # Mock AI services for consistent performance
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai, \
             patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude, \
             patch('scrollintel.agents.scroll_ml_engineer.openai') as mock_ml_openai:
            
            # Configure mocks with slight delay to simulate real API calls
            def mock_openai_response(*args, **kwargs):
                time.sleep(0.1)  # Simulate API latency
                return Mock(choices=[Mock(message=Mock(content="Mock response"))])
            
            def mock_claude_response(*args, **kwargs):
                time.sleep(0.1)
                return Mock(content=[Mock(text="Mock response")])
            
            mock_openai.chat.completions.create.side_effect = mock_openai_response
            mock_claude.messages.create.side_effect = mock_claude_response
            mock_ml_openai.chat.completions.create.side_effect = mock_openai_response
            
            # Create concurrent requests
            async def make_request(agent, request_id):
                start_time = time.time()
                try:
                    request = {
                        "prompt": f"Test request {request_id}",
                        "context": {"request_id": request_id}
                    }
                    response = await agent.process_request(request)
                    
                    response_time = time.time() - start_time
                    monitor.record_response_time(response_time)
                    
                    if response['status'] == 'success':
                        monitor.record_success()
                    else:
                        monitor.record_error()
                    
                    return response
                except Exception as e:
                    response_time = time.time() - start_time
                    monitor.record_response_time(response_time)
                    monitor.record_error()
                    return {"status": "error", "error": str(e)}
            
            # Run 50 concurrent requests across agents
            tasks = []
            for i in range(50):
                agent = agents[i % len(agents)]
                tasks.append(make_request(agent, i))
            
            # Execute all requests concurrently
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            monitor.stop_monitoring()
            
            # Analyze performance
            summary = monitor.get_summary()
            
            # Performance assertions
            assert summary['success_rate'] >= 0.95, f"Success rate too low: {summary['success_rate']}"
            assert summary['avg_response_time'] < 2.0, f"Average response time too high: {summary['avg_response_time']}"
            assert summary['max_cpu_usage'] < 90, f"CPU usage too high: {summary['max_cpu_usage']}"
            assert summary['max_memory_usage'] < 90, f"Memory usage too high: {summary['max_memory_usage']}"
            
            print(f"Performance Summary:")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Requests/Second: {len(tasks) / total_time:.2f}")
            print(f"  Success Rate: {summary['success_rate']:.2%}")
            print(f"  Avg Response Time: {summary['avg_response_time']:.3f}s")
            print(f"  Max CPU Usage: {summary['max_cpu_usage']:.1f}%")
            print(f"  Max Memory Usage: {summary['max_memory_usage']:.1f}%")
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing_performance(self, test_client, test_user_token, sample_datasets):
        """Test performance with large datasets"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Create large dataset
        import pandas as pd
        import numpy as np
        
        large_dataset = pd.DataFrame({
            'id': range(10000),
            'feature_1': np.random.randn(10000),
            'feature_2': np.random.randn(10000),
            'feature_3': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
            'target': np.random.choice([0, 1], 10000)
        })
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_dataset.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test file upload performance
            start_time = time.time()
            with open(temp_file, 'rb') as f:
                files = {"file": ("large_dataset.csv", f, "text/csv")}
                response = test_client.post(
                    "/api/v1/files/upload",
                    files=files,
                    headers=test_user_token
                )
            upload_time = time.time() - start_time
            monitor.record_response_time(upload_time)
            
            assert response.status_code == 200
            dataset_id = response.json()['dataset_id']
            
            # Test analysis performance
            analysis_request = {
                "dataset_id": dataset_id,
                "prompt": "Analyze this large dataset",
                "agent_type": "data_scientist"
            }
            
            with patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude:
                mock_claude.messages.create.return_value = Mock(
                    content=[Mock(text="Large dataset analysis complete")]
                )
                
                start_time = time.time()
                response = test_client.post(
                    "/api/v1/agents/process",
                    json=analysis_request,
                    headers=test_user_token
                )
                analysis_time = time.time() - start_time
                monitor.record_response_time(analysis_time)
            
            assert response.status_code == 200
            
            monitor.stop_monitoring()
            summary = monitor.get_summary()
            
            # Performance assertions for large data
            assert upload_time < 30.0, f"Upload time too slow: {upload_time:.2f}s"
            assert analysis_time < 10.0, f"Analysis time too slow: {analysis_time:.2f}s"
            assert summary['max_memory_usage'] < 95, f"Memory usage too high: {summary['max_memory_usage']}"
            
            print(f"Large Dataset Performance:")
            print(f"  Dataset Size: {len(large_dataset)} rows")
            print(f"  Upload Time: {upload_time:.2f}s")
            print(f"  Analysis Time: {analysis_time:.2f}s")
            print(f"  Max Memory Usage: {summary['max_memory_usage']:.1f}%")
            
        finally:
            # Cleanup
            import os
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, agent_registry, mock_ai_services):
        """Test memory usage under sustained load"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup agents
        cto_agent = ScrollCTOAgent()
        agent_registry.register_agent(cto_agent)
        
        # Mock AI service
        with patch('scrollintel.agents.scroll_cto_agent.openai') as mock_openai:
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Mock response with some content to use memory"))]
            )
            
            # Run sustained load for memory testing
            async def sustained_load():
                for i in range(100):
                    request = {
                        "prompt": f"Memory test request {i} with some additional content to use memory",
                        "context": {"data": list(range(100))}  # Add some data to use memory
                    }
                    
                    try:
                        response = await cto_agent.process_request(request)
                        monitor.record_success()
                    except Exception:
                        monitor.record_error()
                    
                    # Small delay to allow monitoring
                    await asyncio.sleep(0.01)
            
            # Run multiple sustained loads concurrently
            tasks = [sustained_load() for _ in range(5)]
            await asyncio.gather(*tasks)
            
            monitor.stop_monitoring()
            summary = monitor.get_summary()
            
            # Memory usage assertions
            assert summary['max_memory_usage'] < 85, f"Memory usage too high: {summary['max_memory_usage']}"
            assert summary['success_rate'] >= 0.98, f"Success rate too low under load: {summary['success_rate']}"
            
            print(f"Memory Load Test:")
            print(f"  Total Requests: {summary['total_requests']}")
            print(f"  Success Rate: {summary['success_rate']:.2%}")
            print(f"  Max Memory Usage: {summary['max_memory_usage']:.1f}%")
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, agent_registry, mock_ai_services):
        """Test response time consistency across multiple requests"""
        monitor = PerformanceMonitor()
        
        # Setup agent
        ds_agent = ScrollDataScientist()
        agent_registry.register_agent(ds_agent)
        
        response_times = []
        
        with patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude:
            # Add consistent delay to mock
            def mock_response(*args, **kwargs):
                time.sleep(0.1)  # Consistent 100ms delay
                return Mock(content=[Mock(text="Consistent response")])
            
            mock_claude.messages.create.side_effect = mock_response
            
            # Make 20 sequential requests
            for i in range(20):
                request = {
                    "prompt": f"Consistency test {i}",
                    "context": {}
                }
                
                start_time = time.time()
                response = await ds_agent.process_request(request)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                assert response['status'] == 'success'
            
            # Analyze consistency
            avg_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Consistency assertions
            assert std_dev < 0.05, f"Response time too inconsistent: std_dev={std_dev:.3f}"
            assert max_time - min_time < 0.2, f"Response time range too large: {max_time - min_time:.3f}"
            
            print(f"Response Time Consistency:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Std Dev: {std_dev:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            print(f"  Range: {max_time - min_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_file_uploads_performance(self, test_client, test_user_token, temp_files):
        """Test performance with concurrent file uploads"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        async def upload_file(file_path: str, file_id: int):
            start_time = time.time()
            try:
                with open(file_path, 'rb') as f:
                    files = {"file": (f"test_file_{file_id}.csv", f, "text/csv")}
                    response = test_client.post(
                        "/api/v1/files/upload",
                        files=files,
                        headers=test_user_token
                    )
                
                response_time = time.time() - start_time
                monitor.record_response_time(response_time)
                
                if response.status_code == 200:
                    monitor.record_success()
                else:
                    monitor.record_error()
                
                return response.status_code
            except Exception:
                response_time = time.time() - start_time
                monitor.record_response_time(response_time)
                monitor.record_error()
                return 500
        
        # Upload 10 files concurrently
        tasks = [
            upload_file(temp_files['csv_data_csv'], i) 
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        monitor.stop_monitoring()
        summary = monitor.get_summary()
        
        # Performance assertions
        success_count = sum(1 for r in results if r == 200)
        assert success_count >= 8, f"Too many upload failures: {success_count}/10"
        assert summary['avg_response_time'] < 5.0, f"Upload time too slow: {summary['avg_response_time']:.2f}s"
        
        print(f"Concurrent Upload Performance:")
        print(f"  Successful Uploads: {success_count}/10")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Avg Response Time: {summary['avg_response_time']:.2f}s")
        print(f"  Uploads/Second: {len(tasks) / total_time:.2f}")
    
    @pytest.mark.asyncio
    async def test_stress_test_agent_registry(self, agent_registry):
        """Stress test the agent registry with many operations"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Create many agents
        agents = []
        for i in range(100):
            agent = ScrollCTOAgent()
            agent.agent_id = f"test_agent_{i}"
            agents.append(agent)
        
        # Register all agents
        start_time = time.time()
        for agent in agents:
            agent_registry.register_agent(agent)
        registration_time = time.time() - start_time
        
        # Test concurrent lookups
        async def lookup_agent(agent_id: str):
            try:
                agent = agent_registry.get_agent(agent_id)
                return agent is not None
            except Exception:
                return False
        
        # Perform many concurrent lookups
        lookup_tasks = [
            lookup_agent(f"test_agent_{i % 100}") 
            for i in range(1000)
        ]
        
        start_time = time.time()
        lookup_results = await asyncio.gather(*lookup_tasks)
        lookup_time = time.time() - start_time
        
        monitor.stop_monitoring()
        summary = monitor.get_summary()
        
        # Performance assertions
        successful_lookups = sum(lookup_results)
        assert successful_lookups >= 950, f"Too many lookup failures: {successful_lookups}/1000"
        assert registration_time < 1.0, f"Registration too slow: {registration_time:.2f}s"
        assert lookup_time < 2.0, f"Lookups too slow: {lookup_time:.2f}s"
        
        print(f"Agent Registry Stress Test:")
        print(f"  Agents Registered: {len(agents)}")
        print(f"  Registration Time: {registration_time:.3f}s")
        print(f"  Successful Lookups: {successful_lookups}/1000")
        print(f"  Lookup Time: {lookup_time:.3f}s")
        print(f"  Lookups/Second: {1000 / lookup_time:.0f}")


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring fixture"""
    return PerformanceMonitor()