"""
Performance Benchmarks for AGI Memory Architecture

Comprehensive benchmarks testing memory system performance,
scalability, and efficiency under various conditions.
"""

import pytest
import asyncio
import tempfile
import os
import time
import statistics
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from scrollintel.engines.working_memory import WorkingMemorySystem
from scrollintel.engines.long_term_memory import LongTermMemorySystem
from scrollintel.engines.episodic_memory import EpisodicMemorySystem
from scrollintel.engines.memory_integration import MemoryIntegrationSystem
from scrollintel.models.memory_models import MemoryType, MemoryRetrievalQuery


class MemoryBenchmarkSuite:
    """Comprehensive benchmark suite for memory systems"""
    
    def __init__(self):
        self.results = {}
    
    async def run_all_benchmarks(self):
        """Run all memory system benchmarks"""
        print("Starting Memory System Benchmarks...")
        print("=" * 50)
        
        # Working Memory Benchmarks
        await self.benchmark_working_memory()
        
        # Long-term Memory Benchmarks
        await self.benchmark_long_term_memory()
        
        # Episodic Memory Benchmarks
        await self.benchmark_episodic_memory()
        
        # Integration System Benchmarks
        await self.benchmark_integration_system()
        
        # Scalability Benchmarks
        await self.benchmark_scalability()
        
        # Memory Efficiency Benchmarks
        await self.benchmark_memory_efficiency()
        
        # Print summary
        self.print_benchmark_summary()
    
    async def benchmark_working_memory(self):
        """Benchmark working memory operations"""
        print("\n1. Working Memory Benchmarks")
        print("-" * 30)
        
        wm = WorkingMemorySystem(capacity=20, decay_rate=0.01)
        await wm.start()
        
        try:
            # Storage benchmark
            storage_times = []
            for batch in range(10):
                start_time = time.time()
                for i in range(100):
                    await wm.store(f"content_{batch}_{i}", importance=0.5)
                storage_times.append(time.time() - start_time)
            
            avg_storage_time = statistics.mean(storage_times)
            storage_throughput = 100 / avg_storage_time
            
            # Retrieval benchmark
            retrieval_times = []
            for batch in range(10):
                start_time = time.time()
                for i in range(wm.capacity):
                    wm.retrieve(i)
                retrieval_times.append(time.time() - start_time)
            
            avg_retrieval_time = statistics.mean(retrieval_times)
            retrieval_throughput = wm.capacity / avg_retrieval_time
            
            # Search benchmark
            search_times = []
            for batch in range(10):
                start_time = time.time()
                for i in range(50):
                    wm.search("content", similarity_threshold=0.3)
                search_times.append(time.time() - start_time)
            
            avg_search_time = statistics.mean(search_times)
            search_throughput = 50 / avg_search_time
            
            # Concurrent access benchmark
            concurrent_times = []
            for batch in range(5):
                start_time = time.time()
                tasks = []
                for i in range(20):
                    tasks.append(wm.store(f"concurrent_{batch}_{i}", importance=0.5))
                await asyncio.gather(*tasks)
                concurrent_times.append(time.time() - start_time)
            
            avg_concurrent_time = statistics.mean(concurrent_times)
            concurrent_throughput = 20 / avg_concurrent_time
            
            self.results['working_memory'] = {
                'storage_throughput': storage_throughput,
                'retrieval_throughput': retrieval_throughput,
                'search_throughput': search_throughput,
                'concurrent_throughput': concurrent_throughput,
                'avg_storage_time': avg_storage_time,
                'avg_retrieval_time': avg_retrieval_time,
                'avg_search_time': avg_search_time
            }
            
            print(f"  Storage Throughput: {storage_throughput:.1f} ops/sec")
            print(f"  Retrieval Throughput: {retrieval_throughput:.1f} ops/sec")
            print(f"  Search Throughput: {search_throughput:.1f} ops/sec")
            print(f"  Concurrent Throughput: {concurrent_throughput:.1f} ops/sec")
            
        finally:
            await wm.stop()
    
    async def benchmark_long_term_memory(self):
        """Benchmark long-term memory operations"""
        print("\n2. Long-term Memory Benchmarks")
        print("-" * 30)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        ltm = LongTermMemorySystem(db_path=db_path)
        await ltm.start()
        
        try:
            # Storage benchmark
            storage_times = []
            memory_ids = []
            
            for batch in range(5):
                start_time = time.time()
                batch_ids = []
                for i in range(100):
                    memory_id = await ltm.store_memory(
                        content=f"knowledge_{batch}_{i}",
                        memory_type=MemoryType.SEMANTIC,
                        importance=0.5 + (i % 5) * 0.1
                    )
                    batch_ids.append(memory_id)
                storage_times.append(time.time() - start_time)
                memory_ids.extend(batch_ids)
            
            avg_storage_time = statistics.mean(storage_times)
            storage_throughput = 100 / avg_storage_time
            
            # Retrieval benchmark
            retrieval_times = []
            for batch in range(5):
                start_time = time.time()
                for i in range(50):
                    memory_id = memory_ids[batch * 50 + i] if batch * 50 + i < len(memory_ids) else memory_ids[0]
                    await ltm.retrieve_memory(memory_id)
                retrieval_times.append(time.time() - start_time)
            
            avg_retrieval_time = statistics.mean(retrieval_times)
            retrieval_throughput = 50 / avg_retrieval_time
            
            # Search benchmark
            search_times = []
            for batch in range(5):
                query = MemoryRetrievalQuery()
                query.content_keywords = [f"knowledge_{batch}"]
                query.max_results = 20
                
                start_time = time.time()
                await ltm.search_memories(query)
                search_times.append(time.time() - start_time)
            
            avg_search_time = statistics.mean(search_times)
            search_throughput = 1 / avg_search_time
            
            # Association benchmark
            association_times = []
            for batch in range(3):
                start_time = time.time()
                for i in range(20):
                    if i + 1 < len(memory_ids):
                        await ltm.create_association(
                            memory_ids[i], memory_ids[i + 1], 
                            strength=0.5, link_type="semantic"
                        )
                association_times.append(time.time() - start_time)
            
            avg_association_time = statistics.mean(association_times)
            association_throughput = 20 / avg_association_time
            
            self.results['long_term_memory'] = {
                'storage_throughput': storage_throughput,
                'retrieval_throughput': retrieval_throughput,
                'search_throughput': search_throughput,
                'association_throughput': association_throughput,
                'total_memories': len(memory_ids)
            }
            
            print(f"  Storage Throughput: {storage_throughput:.1f} ops/sec")
            print(f"  Retrieval Throughput: {retrieval_throughput:.1f} ops/sec")
            print(f"  Search Throughput: {search_throughput:.1f} ops/sec")
            print(f"  Association Throughput: {association_throughput:.1f} ops/sec")
            print(f"  Total Memories Stored: {len(memory_ids)}")
            
        finally:
            await ltm.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    async def benchmark_episodic_memory(self):
        """Benchmark episodic memory operations"""
        print("\n3. Episodic Memory Benchmarks")
        print("-" * 30)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        em = EpisodicMemorySystem(db_path=db_path)
        await em.start()
        
        try:
            # Episode storage benchmark
            storage_times = []
            episode_ids = []
            
            for batch in range(5):
                start_time = time.time()
                batch_ids = []
                for i in range(50):
                    episode_id = await em.store_episode(
                        event_description=f"Event {batch}_{i}: Learning session",
                        location=f"location_{batch % 3}",
                        participants=[f"person_{i % 5}", f"person_{(i+1) % 5}"],
                        emotions={"interest": 0.7, "satisfaction": 0.6},
                        outcome=f"Learned concept {i}",
                        lessons_learned=[f"Lesson {i}: Important insight"]
                    )
                    batch_ids.append(episode_id)
                storage_times.append(time.time() - start_time)
                episode_ids.extend(batch_ids)
            
            avg_storage_time = statistics.mean(storage_times)
            storage_throughput = 50 / avg_storage_time
            
            # Episode retrieval benchmark
            retrieval_times = []
            for batch in range(5):
                start_time = time.time()
                for i in range(25):
                    episode_id = episode_ids[batch * 25 + i] if batch * 25 + i < len(episode_ids) else episode_ids[0]
                    await em.retrieve_episode(episode_id)
                retrieval_times.append(time.time() - start_time)
            
            avg_retrieval_time = statistics.mean(retrieval_times)
            retrieval_throughput = 25 / avg_retrieval_time
            
            # Episode search benchmark
            search_times = []
            for batch in range(5):
                start_time = time.time()
                await em.search_episodes(
                    query="Learning session",
                    max_results=10
                )
                search_times.append(time.time() - start_time)
            
            avg_search_time = statistics.mean(search_times)
            search_throughput = 1 / avg_search_time
            
            # Temporal sequence benchmark
            sequence_times = []
            for batch in range(3):
                start_time = time.time()
                episode_id = episode_ids[batch * 10] if batch * 10 < len(episode_ids) else episode_ids[0]
                await em.get_temporal_sequence(episode_id, window_size=5)
                sequence_times.append(time.time() - start_time)
            
            avg_sequence_time = statistics.mean(sequence_times)
            sequence_throughput = 1 / avg_sequence_time
            
            self.results['episodic_memory'] = {
                'storage_throughput': storage_throughput,
                'retrieval_throughput': retrieval_throughput,
                'search_throughput': search_throughput,
                'sequence_throughput': sequence_throughput,
                'total_episodes': len(episode_ids)
            }
            
            print(f"  Storage Throughput: {storage_throughput:.1f} ops/sec")
            print(f"  Retrieval Throughput: {retrieval_throughput:.1f} ops/sec")
            print(f"  Search Throughput: {search_throughput:.1f} ops/sec")
            print(f"  Sequence Throughput: {sequence_throughput:.1f} ops/sec")
            print(f"  Total Episodes Stored: {len(episode_ids)}")
            
        finally:
            await em.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    async def benchmark_integration_system(self):
        """Benchmark integrated memory system"""
        print("\n4. Integration System Benchmarks")
        print("-" * 30)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lt_db = os.path.join(tmpdir, "lt_memory.db")
            ep_db = os.path.join(tmpdir, "ep_memory.db")
            
            system = MemoryIntegrationSystem(
                working_memory_capacity=15,
                lt_memory_db=lt_db,
                episodic_db=ep_db
            )
            await system.start()
            
            try:
                # Experience storage benchmark
                storage_times = []
                for batch in range(5):
                    start_time = time.time()
                    for i in range(30):
                        await system.store_experience(
                            content=f"Experience {batch}_{i}",
                            experience_type="learning",
                            context={"location": f"loc_{batch}", "outcome": "success"},
                            importance=0.5 + (i % 5) * 0.1
                        )
                    storage_times.append(time.time() - start_time)
                
                avg_storage_time = statistics.mean(storage_times)
                storage_throughput = 30 / avg_storage_time
                
                # Memory retrieval benchmark
                retrieval_times = []
                for batch in range(5):
                    start_time = time.time()
                    for i in range(10):
                        await system.retrieve_relevant_memories(
                            query=f"Experience {batch}",
                            max_results=5
                        )
                    retrieval_times.append(time.time() - start_time)
                
                avg_retrieval_time = statistics.mean(retrieval_times)
                retrieval_throughput = 10 / avg_retrieval_time
                
                # Insight generation benchmark
                insight_times = []
                for batch in range(3):
                    start_time = time.time()
                    await system.generate_memory_guided_insight(
                        problem=f"How to solve problem {batch}?",
                        context={"domain": "learning"}
                    )
                    insight_times.append(time.time() - start_time)
                
                avg_insight_time = statistics.mean(insight_times)
                insight_throughput = 1 / avg_insight_time
                
                # Decision making benchmark
                decision_times = []
                for batch in range(3):
                    start_time = time.time()
                    await system.memory_guided_decision(
                        decision_context={"task": f"task_{batch}"},
                        options=[f"option_A_{batch}", f"option_B_{batch}", f"option_C_{batch}"]
                    )
                    decision_times.append(time.time() - start_time)
                
                avg_decision_time = statistics.mean(decision_times)
                decision_throughput = 1 / avg_decision_time
                
                # Creative connections benchmark
                start_time = time.time()
                connections = await system.create_creative_connections(
                    domain="general",
                    novelty_threshold=0.4
                )
                creative_time = time.time() - start_time
                creative_throughput = len(connections) / creative_time if creative_time > 0 else 0
                
                self.results['integration_system'] = {
                    'storage_throughput': storage_throughput,
                    'retrieval_throughput': retrieval_throughput,
                    'insight_throughput': insight_throughput,
                    'decision_throughput': decision_throughput,
                    'creative_throughput': creative_throughput,
                    'creative_connections': len(connections)
                }
                
                print(f"  Experience Storage: {storage_throughput:.1f} ops/sec")
                print(f"  Memory Retrieval: {retrieval_throughput:.1f} ops/sec")
                print(f"  Insight Generation: {insight_throughput:.1f} ops/sec")
                print(f"  Decision Making: {decision_throughput:.1f} ops/sec")
                print(f"  Creative Connections: {len(connections)} generated")
                
            finally:
                await system.stop()
    
    async def benchmark_scalability(self):
        """Benchmark system scalability with increasing load"""
        print("\n5. Scalability Benchmarks")
        print("-" * 30)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lt_db = os.path.join(tmpdir, "scalability_lt.db")
            ep_db = os.path.join(tmpdir, "scalability_ep.db")
            
            system = MemoryIntegrationSystem(
                working_memory_capacity=50,
                lt_memory_db=lt_db,
                episodic_db=ep_db
            )
            await system.start()
            
            try:
                scale_results = {}
                
                # Test different scales
                scales = [10, 50, 100, 500, 1000]
                
                for scale in scales:
                    print(f"  Testing scale: {scale} operations")
                    
                    # Storage scalability
                    start_time = time.time()
                    for i in range(scale):
                        await system.store_experience(
                            content=f"Scalability test {i}",
                            experience_type="learning",
                            importance=0.5
                        )
                    storage_time = time.time() - start_time
                    
                    # Retrieval scalability
                    start_time = time.time()
                    for i in range(min(scale // 10, 50)):  # Limit retrieval tests
                        await system.retrieve_relevant_memories(
                            query=f"Scalability test {i}",
                            max_results=5
                        )
                    retrieval_time = time.time() - start_time
                    
                    scale_results[scale] = {
                        'storage_time': storage_time,
                        'storage_throughput': scale / storage_time,
                        'retrieval_time': retrieval_time,
                        'retrieval_throughput': min(scale // 10, 50) / retrieval_time if retrieval_time > 0 else 0
                    }
                    
                    print(f"    Storage: {scale / storage_time:.1f} ops/sec")
                    print(f"    Retrieval: {min(scale // 10, 50) / retrieval_time:.1f} ops/sec" if retrieval_time > 0 else "    Retrieval: N/A")
                
                self.results['scalability'] = scale_results
                
                # Analyze scalability trends
                storage_throughputs = [scale_results[scale]['storage_throughput'] for scale in scales]
                print(f"  Storage throughput trend: {storage_throughputs}")
                
            finally:
                await system.stop()
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory usage and efficiency"""
        print("\n6. Memory Efficiency Benchmarks")
        print("-" * 30)
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lt_db = os.path.join(tmpdir, "efficiency_lt.db")
            ep_db = os.path.join(tmpdir, "efficiency_ep.db")
            
            system = MemoryIntegrationSystem(
                working_memory_capacity=100,
                lt_memory_db=lt_db,
                episodic_db=ep_db
            )
            await system.start()
            
            try:
                # Measure memory usage during operations
                memory_measurements = []
                
                for batch in range(10):
                    # Store batch of experiences
                    for i in range(100):
                        await system.store_experience(
                            content=f"Efficiency test {batch}_{i}" * 10,  # Larger content
                            experience_type="learning",
                            context={"batch": batch, "index": i},
                            importance=0.5
                        )
                    
                    # Measure memory
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_measurements.append(current_memory - initial_memory)
                    
                    # Force garbage collection
                    gc.collect()
                
                # Calculate efficiency metrics
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = final_memory - initial_memory
                
                # Get system metrics
                metrics = system.get_system_metrics()
                total_memories = sum(m.total_memories for m in metrics.values() if hasattr(m, 'total_memories'))
                
                memory_per_item = memory_growth / total_memories if total_memories > 0 else 0
                
                self.results['memory_efficiency'] = {
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_growth_mb': memory_growth,
                    'total_memories': total_memories,
                    'memory_per_item_kb': memory_per_item * 1024,
                    'memory_measurements': memory_measurements
                }
                
                print(f"  Initial Memory: {initial_memory:.1f} MB")
                print(f"  Final Memory: {final_memory:.1f} MB")
                print(f"  Memory Growth: {memory_growth:.1f} MB")
                print(f"  Total Memories: {total_memories}")
                print(f"  Memory per Item: {memory_per_item * 1024:.2f} KB")
                
            finally:
                await system.stop()
    
    def print_benchmark_summary(self):
        """Print comprehensive benchmark summary"""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        if 'working_memory' in self.results:
            wm = self.results['working_memory']
            print(f"\nWorking Memory:")
            print(f"  Best Performance: {max(wm['storage_throughput'], wm['retrieval_throughput']):.1f} ops/sec")
            print(f"  Concurrent Capability: {wm['concurrent_throughput']:.1f} ops/sec")
        
        if 'long_term_memory' in self.results:
            ltm = self.results['long_term_memory']
            print(f"\nLong-term Memory:")
            print(f"  Storage Capacity: {ltm['total_memories']} memories")
            print(f"  Search Performance: {ltm['search_throughput']:.1f} ops/sec")
        
        if 'episodic_memory' in self.results:
            em = self.results['episodic_memory']
            print(f"\nEpisodic Memory:")
            print(f"  Episode Capacity: {em['total_episodes']} episodes")
            print(f"  Temporal Queries: {em['sequence_throughput']:.1f} ops/sec")
        
        if 'integration_system' in self.results:
            integration = self.results['integration_system']
            print(f"\nIntegration System:")
            print(f"  End-to-end Performance: {integration['storage_throughput']:.1f} ops/sec")
            print(f"  Insight Generation: {integration['insight_throughput']:.1f} ops/sec")
            print(f"  Creative Connections: {integration['creative_connections']} generated")
        
        if 'memory_efficiency' in self.results:
            efficiency = self.results['memory_efficiency']
            print(f"\nMemory Efficiency:")
            print(f"  Memory per Item: {efficiency['memory_per_item_kb']:.2f} KB")
            print(f"  Total Growth: {efficiency['memory_growth_mb']:.1f} MB")
        
        print("\n" + "=" * 50)


@pytest.mark.asyncio
async def test_comprehensive_benchmarks():
    """Run comprehensive memory system benchmarks"""
    benchmark_suite = MemoryBenchmarkSuite()
    await benchmark_suite.run_all_benchmarks()
    
    # Verify benchmark results meet minimum performance requirements
    results = benchmark_suite.results
    
    # Working memory should handle at least 100 ops/sec for basic operations
    if 'working_memory' in results:
        assert results['working_memory']['storage_throughput'] > 50
        assert results['working_memory']['retrieval_throughput'] > 100
    
    # Long-term memory should handle reasonable loads
    if 'long_term_memory' in results:
        assert results['long_term_memory']['total_memories'] >= 500
        assert results['long_term_memory']['search_throughput'] > 1
    
    # Integration system should provide end-to-end functionality
    if 'integration_system' in results:
        assert results['integration_system']['storage_throughput'] > 10
        assert results['integration_system']['insight_throughput'] > 0.5
    
    # Memory efficiency should be reasonable
    if 'memory_efficiency' in results:
        assert results['memory_efficiency']['memory_per_item_kb'] < 100  # Less than 100KB per item


@pytest.mark.asyncio
async def test_stress_testing():
    """Stress test memory systems under high load"""
    print("\nStress Testing Memory Systems...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        lt_db = os.path.join(tmpdir, "stress_lt.db")
        ep_db = os.path.join(tmpdir, "stress_ep.db")
        
        system = MemoryIntegrationSystem(
            working_memory_capacity=20,
            lt_memory_db=lt_db,
            episodic_db=ep_db
        )
        await system.start()
        
        try:
            # Concurrent stress test
            async def stress_worker(worker_id, operations):
                for i in range(operations):
                    await system.store_experience(
                        content=f"Stress test worker {worker_id} operation {i}",
                        experience_type="learning",
                        importance=0.5
                    )
                    
                    if i % 10 == 0:  # Occasional retrieval
                        await system.retrieve_relevant_memories(
                            query=f"worker {worker_id}",
                            max_results=3
                        )
            
            # Run multiple workers concurrently
            start_time = time.time()
            workers = [stress_worker(i, 50) for i in range(10)]
            await asyncio.gather(*workers)
            stress_time = time.time() - start_time
            
            total_operations = 10 * 50  # 10 workers * 50 operations each
            stress_throughput = total_operations / stress_time
            
            print(f"Stress Test Results:")
            print(f"  Concurrent Workers: 10")
            print(f"  Total Operations: {total_operations}")
            print(f"  Time Taken: {stress_time:.2f} seconds")
            print(f"  Throughput: {stress_throughput:.1f} ops/sec")
            
            # Verify system still functions correctly
            memories = await system.retrieve_relevant_memories("stress test", max_results=5)
            assert len(memories['long_term_memory']) > 0 or len(memories['working_memory']) > 0
            
            # Performance should be reasonable even under stress
            assert stress_throughput > 20  # At least 20 ops/sec under stress
            
        finally:
            await system.stop()


if __name__ == "__main__":
    # Run benchmarks directly
    asyncio.run(MemoryBenchmarkSuite().run_all_benchmarks())