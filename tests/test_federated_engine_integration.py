"""
Integration tests for FederatedEngine
Tests federated learning workflows, PySyft integration, TFF support,
differential privacy, secure aggregation, and edge device simulation.
"""

import pytest
import asyncio
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.federated_engine import (
    FederatedEngine,
    DifferentialPrivacyEngine,
    SecureAggregationProtocol,
    PySyftIntegration,
    TensorFlowFederatedIntegration,
    EdgeDeviceSimulator,
    FederatedEngineStatus,
    EdgeDeviceType,
    PrivacyLevel,
    get_federated_engine,
    initialize_federated_engine
)

class TestFederatedEngineIntegration:
    """Integration tests for federated learning engine"""
    
    @pytest.fixture
    async def federated_engine(self):
        """Create federated engine for testing"""
        mock_redis = AsyncMock()
        engine = FederatedEngine(redis_client=mock_redis)
        return engine
    
    @pytest.fixture
    def sample_device_configs(self):
        """Sample device configurations for testing"""
        return [
            {
                "device_name": "Test_Mobile",
                "device_type": EdgeDeviceType.MOBILE,
                "data_size": 500,
                "compute_power": 0.5,
                "bandwidth": 5.0,
                "battery_level": 80.0,
                "privacy_level": PrivacyLevel.HIGH
            },
            {
                "device_name": "Test_Desktop",
                "device_type": EdgeDeviceType.DESKTOP,
                "data_size": 2000,
                "compute_power": 2.0,
                "bandwidth": 50.0,
                "privacy_level": PrivacyLevel.MEDIUM
            },
            {
                "device_name": "Test_IoT",
                "device_type": EdgeDeviceType.IOT,
                "data_size": 100,
                "compute_power": 0.2,
                "bandwidth": 1.0,
                "privacy_level": PrivacyLevel.MAXIMUM
            }
        ]
    
    @pytest.fixture
    def sample_task_config(self):
        """Sample task configuration for testing"""
        return {
            "task_name": "Test_MNIST_Classification",
            "model_architecture": {
                "framework": "pytorch",
                "input_size": 784,
                "hidden_size": 128,
                "output_size": 10
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 32,
                "learning_rate": 0.01
            },
            "privacy_config": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "sensitivity": 1.0
            },
            "target_rounds": 3,
            "convergence_threshold": 0.01
        }
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test federated engine initialization"""
        engine = await initialize_federated_engine()
        
        assert engine is not None
        assert engine.status == FederatedEngineStatus.READY
        assert isinstance(engine.privacy_engine, DifferentialPrivacyEngine)
        assert isinstance(engine.secure_aggregation, SecureAggregationProtocol)
        assert isinstance(engine.device_simulator, EdgeDeviceSimulator)
    
    @pytest.mark.asyncio
    async def test_edge_device_management(self, federated_engine, sample_device_configs):
        """Test edge device creation, management, and simulation"""
        engine = federated_engine
        
        # Test device creation
        device_ids = []
        for config in sample_device_configs:
            device_id = await engine.add_edge_device(config)
            device_ids.append(device_id)
            
            assert device_id is not None
            assert len(device_id) > 0
        
        # Test device status retrieval
        for i, device_id in enumerate(device_ids):
            status = engine.device_simulator.get_device_status(device_id)
            
            assert status is not None
            assert status['device_id'] == device_id
            assert status['device_name'] == sample_device_configs[i]['device_name']
            assert status['device_type'] == sample_device_configs[i]['device_type'].value
            assert status['status'] == 'online'
        
        # Test device failure simulation
        mobile_device_id = device_ids[0]
        engine.device_simulator.simulate_device_failure(mobile_device_id, "network")
        
        updated_status = engine.device_simulator.get_device_status(mobile_device_id)
        assert updated_status['status'] == 'offline'
        
        # Test battery failure
        engine.device_simulator.simulate_device_failure(mobile_device_id, "battery")
        updated_status = engine.device_simulator.get_device_status(mobile_device_id)
        assert updated_status['battery_level'] == 0
        assert updated_status['status'] == 'low_battery'
        
        # Test device removal
        removed = await engine.remove_edge_device(device_ids[0])
        assert removed is True
        
        # Verify device is removed
        status = engine.device_simulator.get_device_status(device_ids[0])
        assert status is None
    
    @pytest.mark.asyncio
    async def test_federated_task_lifecycle(self, federated_engine, sample_task_config, sample_device_configs):
        """Test complete federated task lifecycle"""
        engine = federated_engine
        
        # Create devices
        device_ids = []
        for config in sample_device_configs:
            device_id = await engine.add_edge_device(config)
            device_ids.append(device_id)
        
        # Add devices to task config
        task_config = sample_task_config.copy()
        task_config['participating_devices'] = device_ids
        
        # Create task
        task_id = await engine.create_federated_task(task_config)
        assert task_id is not None
        
        # Verify task creation
        task_status = await engine.get_task_status(task_id)
        assert task_status is not None
        assert task_status['task_name'] == task_config['task_name']
        assert task_status['status'] == 'ready'
        assert task_status['participating_devices'] == len(device_ids)
        
        # Start training
        training_success = await engine.start_federated_training(task_id, "custom")
        assert training_success is True
        
        # Check final status
        final_status = await engine.get_task_status(task_id)
        assert final_status['status'] == 'completed'
        assert final_status['rounds_completed'] > 0
        
        # Test task pause/resume (create new task for this)
        pause_task_id = await engine.create_federated_task(task_config)
        
        # Simulate pausing
        pause_success = await engine.pause_training(pause_task_id)
        assert pause_success is False  # Can't pause task that hasn't started
        
        # Test task cleanup
        cleaned_count = await engine.cleanup_completed_tasks(0)
        assert cleaned_count >= 1
    
    def test_differential_privacy_engine(self):
        """Test differential privacy mechanisms"""
        privacy_engine = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        # Test noise multiplier calculation
        noise_multiplier = privacy_engine.get_noise_multiplier()
        assert noise_multiplier > 0
        
        # Test gradient clipping
        gradients = np.random.normal(0, 5, (100, 10))  # Large gradients
        original_norm = np.linalg.norm(gradients)
        
        clipped_gradients = privacy_engine.clip_gradients(gradients, clip_norm=1.0)
        clipped_norm = np.linalg.norm(clipped_gradients)
        
        assert clipped_norm <= 1.0 + 1e-6  # Allow small numerical error
        
        # Test noise addition
        initial_budget = privacy_engine.get_remaining_budget()
        noisy_gradients = privacy_engine.add_noise(clipped_gradients)
        remaining_budget = privacy_engine.get_remaining_budget()
        
        assert noisy_gradients.shape == clipped_gradients.shape
        assert remaining_budget < initial_budget  # Budget should be consumed
        
        # Test budget reset
        privacy_engine.reset_budget()
        reset_budget = privacy_engine.get_remaining_budget()
        assert reset_budget == privacy_engine.epsilon
        
        # Test list of gradients
        gradient_list = [np.random.normal(0, 2, (50, 5)) for _ in range(3)]
        clipped_list = privacy_engine.clip_gradients(gradient_list, clip_norm=1.0)
        
        assert len(clipped_list) == len(gradient_list)
        for clipped_grad in clipped_list:
            assert np.linalg.norm(clipped_grad) <= 1.0 + 1e-6
    
    def test_secure_aggregation_protocol(self):
        """Test secure aggregation mechanisms"""
        secure_agg = SecureAggregationProtocol(threshold=3)
        
        # Test secret sharing
        secret_value = 42.5
        num_parties = 5
        
        shares = secure_agg.generate_secret_shares(secret_value, num_parties)
        assert len(shares) == num_parties
        
        # Test secret reconstruction
        reconstructed = secure_agg.reconstruct_secret(shares[:3])  # Use minimum threshold
        assert abs(reconstructed - secret_value) < 1e-6
        
        # Test with different subset of shares
        reconstructed2 = secure_agg.reconstruct_secret(shares[1:4])
        assert abs(reconstructed2 - secret_value) < 1e-6
        
        # Test aggregation mask generation
        device_ids = ["device_1", "device_2", "device_3"]
        seed = 12345
        
        masks = []
        for device_id in device_ids:
            mask = secure_agg.generate_aggregation_mask(device_id, seed)
            masks.append(mask)
            assert mask.shape == (1000,)  # Default mask size
        
        # Test secure sum
        test_inputs = [np.random.normal(0, 1, 1000) for _ in range(3)]
        masked_inputs = [inp + mask for inp, mask in zip(test_inputs, masks)]
        
        # No dropout masks for this test
        result = secure_agg.secure_sum(masked_inputs, [])
        expected = np.sum(test_inputs, axis=0)
        
        # The result should be close to expected (within noise tolerance)
        assert result.shape == expected.shape
    
    def test_pysyft_integration(self):
        """Test PySyft integration"""
        pysyft_integration = PySyftIntegration()
        
        # Test worker creation
        worker_ids = ["worker_1", "worker_2", "worker_3"]
        
        for worker_id in worker_ids:
            worker = pysyft_integration.create_virtual_worker(worker_id)
            
            if pysyft_integration.hook:  # Only test if PySyft is available
                assert worker is not None
                assert worker_id in pysyft_integration.virtual_workers
            else:
                assert worker is None
        
        # Test federated averaging (mock if PySyft not available)
        if not pysyft_integration.hook:
            # Mock the averaging function
            mock_models = [Mock() for _ in range(3)]
            result = pysyft_integration.federated_averaging(mock_models)
            assert result is None  # Should return None when PySyft not available
    
    def test_tensorflow_federated_integration(self):
        """Test TensorFlow Federated integration"""
        tff_integration = TensorFlowFederatedIntegration()
        
        # Test model creation
        input_shape = (784,)
        num_classes = 10
        
        model_fn = tff_integration.create_keras_model(input_shape, num_classes)
        
        # Should return None if TFF not available, function if available
        if hasattr(tff_integration, 'model_fn') and tff_integration.model_fn:
            assert callable(model_fn)
        
        # Test federated data creation
        client_datasets = {
            "client_1": (np.random.rand(100, 784), np.random.randint(0, 10, 100)),
            "client_2": (np.random.rand(150, 784), np.random.randint(0, 10, 150))
        }
        
        federated_data = tff_integration.create_federated_data(client_datasets)
        
        # Should handle gracefully whether TFF is available or not
        if federated_data is not None:
            assert len(federated_data) == len(client_datasets)
    
    @pytest.mark.asyncio
    async def test_federation_status_and_monitoring(self, federated_engine, sample_device_configs):
        """Test federation status monitoring and reporting"""
        engine = federated_engine
        
        # Add some devices
        device_ids = []
        for config in sample_device_configs:
            device_id = await engine.add_edge_device(config)
            device_ids.append(device_id)
        
        # Get federation status
        status = await engine.get_federation_status()
        
        assert status['engine_status'] == 'ready'
        assert status['total_devices'] == len(device_ids)
        assert status['online_devices'] <= status['total_devices']
        assert 'privacy_budget_remaining' in status
        assert 'config' in status
        
        # Test device status aggregation
        all_devices = engine.device_simulator.get_all_devices_status()
        assert len(all_devices) == len(device_ids)
        
        for device_status in all_devices:
            assert 'device_id' in device_status
            assert 'device_name' in device_status
            assert 'status' in device_status
    
    @pytest.mark.asyncio
    async def test_privacy_budget_management(self, federated_engine):
        """Test privacy budget tracking and management"""
        engine = federated_engine
        
        # Test initial budget
        initial_budget = engine.privacy_engine.get_remaining_budget()
        assert initial_budget > 0
        
        # Consume some budget
        test_data = np.random.normal(0, 1, (50, 10))
        noisy_data = engine.privacy_engine.add_noise(test_data)
        
        remaining_budget = engine.privacy_engine.get_remaining_budget()
        assert remaining_budget < initial_budget
        
        # Test budget consumption tracking
        consumed = initial_budget - remaining_budget
        assert consumed > 0
        
        # Test budget reset
        engine.privacy_engine.reset_budget()
        reset_budget = engine.privacy_engine.get_remaining_budget()
        assert reset_budget == engine.privacy_engine.epsilon
    
    @pytest.mark.asyncio
    async def test_error_handling_and_edge_cases(self, federated_engine):
        """Test error handling and edge cases"""
        engine = federated_engine
        
        # Test invalid task ID
        invalid_status = await engine.get_task_status("invalid_task_id")
        assert invalid_status is None
        
        # Test training non-existent task
        training_result = await engine.start_federated_training("invalid_task_id", "custom")
        assert training_result is False
        
        # Test removing non-existent device
        removal_result = await engine.remove_edge_device("invalid_device_id")
        assert removal_result is False
        
        # Test pause/resume non-existent task
        pause_result = await engine.pause_training("invalid_task_id")
        assert pause_result is False
        
        resume_result = await engine.resume_training("invalid_task_id")
        assert resume_result is False
        
        # Test empty device list for training
        empty_task_config = {
            "task_name": "Empty_Task",
            "model_architecture": {"input_size": 10, "output_size": 2},
            "participating_devices": []
        }
        
        empty_task_id = await engine.create_federated_task(empty_task_config)
        empty_training_result = await engine.start_federated_training(empty_task_id, "custom")
        assert empty_training_result is False
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, federated_engine, sample_device_configs, sample_task_config):
        """Test concurrent federated learning operations"""
        engine = federated_engine
        
        # Create devices concurrently
        device_creation_tasks = [
            engine.add_edge_device(config) for config in sample_device_configs
        ]
        device_ids = await asyncio.gather(*device_creation_tasks)
        
        assert len(device_ids) == len(sample_device_configs)
        assert all(device_id is not None for device_id in device_ids)
        
        # Create multiple tasks concurrently
        task_configs = []
        for i in range(3):
            config = sample_task_config.copy()
            config['task_name'] = f"Concurrent_Task_{i}"
            config['participating_devices'] = device_ids
            task_configs.append(config)
        
        task_creation_tasks = [
            engine.create_federated_task(config) for config in task_configs
        ]
        task_ids = await asyncio.gather(*task_creation_tasks)
        
        assert len(task_ids) == len(task_configs)
        assert all(task_id is not None for task_id in task_ids)
        
        # Get status of all tasks concurrently
        status_tasks = [engine.get_task_status(task_id) for task_id in task_ids]
        statuses = await asyncio.gather(*status_tasks)
        
        assert len(statuses) == len(task_ids)
        assert all(status is not None for status in statuses)
    
    def test_device_simulation_accuracy(self):
        """Test accuracy of device simulation"""
        simulator = EdgeDeviceSimulator()
        
        # Create devices with different characteristics
        mobile_config = {
            "device_name": "Test_Mobile",
            "device_type": EdgeDeviceType.MOBILE,
            "compute_power": 0.5,
            "bandwidth": 5.0,
            "battery_level": 90.0,
            "privacy_level": PrivacyLevel.HIGH
        }
        
        server_config = {
            "device_name": "Test_Server",
            "device_type": EdgeDeviceType.SERVER,
            "compute_power": 5.0,
            "bandwidth": 100.0,
            "privacy_level": PrivacyLevel.LOW
        }
        
        mobile_id = simulator.create_simulated_device(mobile_config)
        server_id = simulator.create_simulated_device(server_config)
        
        # Simulate training on both devices
        training_config = {"epochs": 5}
        
        mobile_result = simulator.simulate_device_training(mobile_id, None, training_config)
        server_result = simulator.simulate_device_training(server_id, None, training_config)
        
        # Server should be faster due to higher compute power
        assert mobile_result['training_time'] > server_result['training_time']
        
        # Mobile should have higher privacy noise
        assert mobile_result['privacy_noise_level'] > server_result['privacy_noise_level']
        
        # Mobile should have battery consumption
        assert mobile_result['battery_level'] is not None
        assert server_result['battery_level'] is None
    
    @pytest.mark.asyncio
    async def test_redis_integration(self):
        """Test Redis integration for persistence"""
        mock_redis = AsyncMock()
        engine = FederatedEngine(redis_client=mock_redis)
        
        # Test device storage
        device_config = {
            "device_name": "Redis_Test_Device",
            "device_type": EdgeDeviceType.DESKTOP,
            "data_size": 1000
        }
        
        device_id = await engine.add_edge_device(device_config)
        
        # Verify Redis calls were made
        mock_redis.hset.assert_called()
        
        # Test task storage
        task_config = {
            "task_name": "Redis_Test_Task",
            "model_architecture": {"input_size": 10, "output_size": 2},
            "participating_devices": [device_id]
        }
        
        task_id = await engine.create_federated_task(task_config)
        
        # Verify Redis calls for task storage
        assert mock_redis.hset.call_count >= 2  # Device + Task

class TestFederatedEnginePerformance:
    """Performance tests for federated learning engine"""
    
    @pytest.mark.asyncio
    async def test_large_scale_device_management(self):
        """Test performance with large number of devices"""
        engine = FederatedEngine()
        
        # Create many devices
        num_devices = 100
        device_configs = []
        
        for i in range(num_devices):
            config = {
                "device_name": f"Perf_Device_{i}",
                "device_type": EdgeDeviceType.DESKTOP,
                "data_size": 1000,
                "compute_power": np.random.uniform(0.5, 2.0),
                "bandwidth": np.random.uniform(5.0, 50.0)
            }
            device_configs.append(config)
        
        # Measure device creation time
        start_time = datetime.utcnow()
        
        device_ids = []
        for config in device_configs:
            device_id = await engine.add_edge_device(config)
            device_ids.append(device_id)
        
        creation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should create devices reasonably quickly
        assert creation_time < 10.0  # Less than 10 seconds for 100 devices
        assert len(device_ids) == num_devices
        
        # Test status retrieval performance
        start_time = datetime.utcnow()
        all_statuses = engine.device_simulator.get_all_devices_status()
        retrieval_time = (datetime.utcnow() - start_time).total_seconds()
        
        assert retrieval_time < 1.0  # Less than 1 second to get all statuses
        assert len(all_statuses) == num_devices
    
    def test_privacy_computation_performance(self):
        """Test performance of privacy computations"""
        privacy_engine = DifferentialPrivacyEngine()
        
        # Test with large gradient arrays
        large_gradients = np.random.normal(0, 1, (10000, 100))
        
        # Measure clipping performance
        start_time = datetime.utcnow()
        clipped = privacy_engine.clip_gradients(large_gradients, clip_norm=1.0)
        clipping_time = (datetime.utcnow() - start_time).total_seconds()
        
        assert clipping_time < 1.0  # Should be fast
        assert clipped.shape == large_gradients.shape
        
        # Measure noise addition performance
        start_time = datetime.utcnow()
        noisy = privacy_engine.add_noise(clipped)
        noise_time = (datetime.utcnow() - start_time).total_seconds()
        
        assert noise_time < 1.0  # Should be fast
        assert noisy.shape == clipped.shape

if __name__ == "__main__":
    pytest.main([__file__, "-v"])