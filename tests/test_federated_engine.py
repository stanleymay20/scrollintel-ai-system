"""
Unit tests for FederatedEngine components
Tests individual components of the federated learning system.
"""

import pytest
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.federated_engine import (
    DifferentialPrivacyEngine,
    SecureAggregationProtocol,
    PySyftIntegration,
    TensorFlowFederatedIntegration,
    EdgeDeviceSimulator,
    FederatedEngine,
    FederatedEngineStatus,
    EdgeDeviceType,
    PrivacyLevel
)

class TestDifferentialPrivacyEngine:
    """Unit tests for differential privacy engine"""
    
    def test_initialization(self):
        """Test privacy engine initialization"""
        engine = DifferentialPrivacyEngine(epsilon=2.0, delta=1e-6, sensitivity=0.5)
        
        assert engine.epsilon == 2.0
        assert engine.delta == 1e-6
        assert engine.sensitivity == 0.5
        assert engine.privacy_budget_used == 0.0
        assert engine.mechanism == "gaussian"
    
    def test_noise_multiplier_calculation(self):
        """Test noise multiplier calculation for different mechanisms"""
        # Gaussian mechanism
        gaussian_engine = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, mechanism="gaussian")
        gaussian_multiplier = gaussian_engine.get_noise_multiplier()
        assert gaussian_multiplier > 0
        
        # Laplace mechanism
        laplace_engine = DifferentialPrivacyEngine(epsilon=1.0, mechanism="laplace")
        laplace_multiplier = laplace_engine.get_noise_multiplier()
        assert laplace_multiplier > 0
        
        # Invalid mechanism
        invalid_engine = DifferentialPrivacyEngine(mechanism="invalid")
        with pytest.raises(ValueError):
            invalid_engine.get_noise_multiplier()
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        engine = DifferentialPrivacyEngine()
        
        # Test single gradient array
        gradients = np.array([[3.0, 4.0], [1.0, 2.0]])  # Norm = 5.477
        clipped = engine.clip_gradients(gradients, clip_norm=1.0)
        
        clipped_norm = np.linalg.norm(clipped)
        assert clipped_norm <= 1.0 + 1e-6
        
        # Test list of gradient arrays
        gradient_list = [
            np.array([3.0, 4.0]),  # Norm = 5.0
            np.array([1.0, 1.0]),  # Norm = 1.414
            np.array([0.5, 0.5])   # Norm = 0.707
        ]
        
        clipped_list = engine.clip_gradients(gradient_list, clip_norm=1.0)
        
        assert len(clipped_list) == len(gradient_list)
        for clipped_grad in clipped_list:
            assert np.linalg.norm(clipped_grad) <= 1.0 + 1e-6
    
    def test_noise_addition(self):
        """Test differential privacy noise addition"""
        engine = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        # Test Gaussian noise
        data = np.zeros((10, 5))
        noisy_data = engine.add_noise(data)
        
        assert noisy_data.shape == data.shape
        assert not np.array_equal(data, noisy_data)  # Should be different due to noise
        assert engine.privacy_budget_used > 0
        
        # Test Laplace noise
        laplace_engine = DifferentialPrivacyEngine(epsilon=1.0, mechanism="laplace")
        laplace_noisy = laplace_engine.add_noise(data.copy())
        
        assert laplace_noisy.shape == data.shape
        assert not np.array_equal(data, laplace_noisy)
        
        # Test with custom sensitivity
        custom_noisy = engine.add_noise(data.copy(), sensitivity=2.0)
        assert custom_noisy.shape == data.shape
    
    def test_privacy_budget_management(self):
        """Test privacy budget tracking and management"""
        engine = DifferentialPrivacyEngine(epsilon=2.0)
        
        # Initial budget
        assert engine.get_remaining_budget() == 2.0
        
        # Consume budget
        data = np.random.normal(0, 1, (5, 3))
        engine.add_noise(data)
        
        remaining = engine.get_remaining_budget()
        assert remaining < 2.0
        assert remaining >= 0
        
        # Reset budget
        engine.reset_budget()
        assert engine.get_remaining_budget() == 2.0
        assert engine.privacy_budget_used == 0.0

class TestSecureAggregationProtocol:
    """Unit tests for secure aggregation protocol"""
    
    def test_initialization(self):
        """Test secure aggregation initialization"""
        protocol = SecureAggregationProtocol(threshold=3)
        
        assert protocol.threshold == 3
        assert protocol.secret_shares == {}
        assert protocol.reconstruction_shares == {}
        assert protocol.aggregation_masks == {}
    
    def test_secret_sharing(self):
        """Test Shamir's secret sharing"""
        protocol = SecureAggregationProtocol(threshold=3)
        
        secret = 42.5
        num_parties = 5
        
        shares = protocol.generate_secret_shares(secret, num_parties)
        
        assert len(shares) == num_parties
        assert all(isinstance(share, tuple) and len(share) == 2 for share in shares)
        
        # Test with insufficient parties
        with pytest.raises(ValueError):
            protocol.generate_secret_shares(secret, 2)  # Less than threshold
    
    def test_secret_reconstruction(self):
        """Test secret reconstruction from shares"""
        protocol = SecureAggregationProtocol(threshold=3)
        
        secret = 123.456
        shares = protocol.generate_secret_shares(secret, 5)
        
        # Reconstruct with minimum threshold
        reconstructed = protocol.reconstruct_secret(shares[:3])
        assert abs(reconstructed - secret) < 1e-10
        
        # Reconstruct with more shares
        reconstructed2 = protocol.reconstruct_secret(shares[:4])
        assert abs(reconstructed2 - secret) < 1e-10
        
        # Test with different subset
        reconstructed3 = protocol.reconstruct_secret([shares[0], shares[2], shares[4]])
        assert abs(reconstructed3 - secret) < 1e-10
        
        # Test with insufficient shares
        with pytest.raises(ValueError):
            protocol.reconstruct_secret(shares[:2])
    
    def test_aggregation_mask_generation(self):
        """Test pseudorandom mask generation"""
        protocol = SecureAggregationProtocol()
        
        device_ids = ["device_1", "device_2", "device_3"]
        seed = 12345
        
        masks = []
        for device_id in device_ids:
            mask = protocol.generate_aggregation_mask(device_id, seed)
            masks.append(mask)
            
            assert mask.shape == (1000,)  # Default size
            assert device_id in protocol.aggregation_masks
        
        # Masks should be different for different devices
        assert not np.array_equal(masks[0], masks[1])
        assert not np.array_equal(masks[1], masks[2])
        
        # Same device with same seed should produce same mask
        mask_repeat = protocol.generate_aggregation_mask(device_ids[0], seed)
        assert np.array_equal(masks[0], mask_repeat)
    
    def test_secure_sum(self):
        """Test secure sum computation"""
        protocol = SecureAggregationProtocol()
        
        # Test basic secure sum
        inputs = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        
        result = protocol.secure_sum(inputs, [])
        expected = np.array([12.0, 15.0, 18.0])
        
        assert np.allclose(result, expected)
        
        # Test with dropout masks
        dropout_masks = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        result_with_dropout = protocol.secure_sum(inputs, dropout_masks)
        expected_with_dropout = expected - np.sum(dropout_masks, axis=0)
        
        assert np.allclose(result_with_dropout, expected_with_dropout)
        
        # Test with empty inputs
        empty_result = protocol.secure_sum([], [])
        assert empty_result.size == 0

class TestEdgeDeviceSimulator:
    """Unit tests for edge device simulator"""
    
    def test_device_creation(self):
        """Test simulated device creation"""
        simulator = EdgeDeviceSimulator()
        
        device_config = {
            "device_name": "Test_Device",
            "device_type": EdgeDeviceType.MOBILE,
            "data_size": 1000,
            "compute_power": 0.8,
            "bandwidth": 15.0,
            "battery_level": 75.0,
            "privacy_level": PrivacyLevel.HIGH
        }
        
        device_id = simulator.create_simulated_device(device_config)
        
        assert device_id is not None
        assert device_id in simulator.devices
        assert device_id in simulator.device_data
        
        # Check device properties
        device = simulator.devices[device_id]
        assert device.device_name == "Test_Device"
        assert device.device_type == EdgeDeviceType.MOBILE
        assert device.data_size == 1000
        assert device.compute_power == 0.8
        assert device.bandwidth == 15.0
        assert device.battery_level == 75.0
        assert device.privacy_level == PrivacyLevel.HIGH
        
        # Check generated data
        device_data = simulator.device_data[device_id]
        assert device_data['data_size'] == 1000
        assert 'features' in device_data
        assert 'labels' in device_data
    
    def test_device_data_generation(self):
        """Test synthetic data generation for different device types"""
        simulator = EdgeDeviceSimulator()
        
        # Mobile device (image data)
        mobile_config = {
            "device_type": EdgeDeviceType.MOBILE,
            "data_size": 100
        }
        mobile_id = simulator.create_simulated_device(mobile_config)
        mobile_data = simulator.device_data[mobile_id]
        
        assert mobile_data['features'].shape == (100, 28, 28)
        assert mobile_data['labels'].shape == (100,)
        assert np.all(mobile_data['labels'] >= 0) and np.all(mobile_data['labels'] < 10)
        
        # IoT device (sensor data)
        iot_config = {
            "device_type": EdgeDeviceType.IOT,
            "data_size": 50
        }
        iot_id = simulator.create_simulated_device(iot_config)
        iot_data = simulator.device_data[iot_id]
        
        assert iot_data['features'].shape == (50, 10)
        assert iot_data['labels'].shape == (50,)
        assert np.all(iot_data['labels'] >= 0) and np.all(iot_data['labels'] < 2)
        
        # Desktop device (tabular data)
        desktop_config = {
            "device_type": EdgeDeviceType.DESKTOP,
            "data_size": 200
        }
        desktop_id = simulator.create_simulated_device(desktop_config)
        desktop_data = simulator.device_data[desktop_id]
        
        assert desktop_data['features'].shape == (200, 20)
        assert desktop_data['labels'].shape == (200,)
        assert np.all(desktop_data['labels'] >= 0) and np.all(desktop_data['labels'] < 5)
    
    def test_device_training_simulation(self):
        """Test training simulation on devices"""
        simulator = EdgeDeviceSimulator()
        
        device_config = {
            "device_name": "Training_Test_Device",
            "device_type": EdgeDeviceType.DESKTOP,
            "compute_power": 2.0,
            "bandwidth": 25.0,
            "battery_level": 80.0,
            "privacy_level": PrivacyLevel.MEDIUM
        }
        
        device_id = simulator.create_simulated_device(device_config)
        
        training_config = {
            "epochs": 10,
            "batch_size": 32
        }
        
        result = simulator.simulate_device_training(device_id, None, training_config)
        
        assert result['device_id'] == device_id
        assert result['training_time'] > 0
        assert result['network_delay'] > 0
        assert 0 <= result['training_accuracy'] <= 1
        assert result['training_loss'] >= 0
        assert result['data_size'] > 0
        assert result['privacy_noise_level'] > 0
        
        # Check battery consumption for mobile device
        mobile_config = device_config.copy()
        mobile_config['device_type'] = EdgeDeviceType.MOBILE
        mobile_id = simulator.create_simulated_device(mobile_config)
        
        mobile_result = simulator.simulate_device_training(mobile_id, None, training_config)
        assert mobile_result['battery_level'] < 80.0  # Should consume battery
    
    def test_device_failure_simulation(self):
        """Test device failure simulation"""
        simulator = EdgeDeviceSimulator()
        
        device_config = {
            "device_type": EdgeDeviceType.MOBILE,
            "battery_level": 50.0,
            "compute_power": 1.0
        }
        
        device_id = simulator.create_simulated_device(device_config)
        device = simulator.devices[device_id]
        
        # Test network failure
        simulator.simulate_device_failure(device_id, "network")
        assert device.status == "offline"
        
        # Test battery failure
        simulator.simulate_device_failure(device_id, "battery")
        assert device.battery_level == 0
        assert device.status == "low_battery"
        
        # Test compute failure
        original_compute = device.compute_power
        simulator.simulate_device_failure(device_id, "compute")
        assert device.compute_power < original_compute
        assert device.status == "degraded"
    
    def test_device_status_retrieval(self):
        """Test device status retrieval"""
        simulator = EdgeDeviceSimulator()
        
        device_config = {
            "device_name": "Status_Test_Device",
            "device_type": EdgeDeviceType.CLOUD,
            "compute_power": 5.0
        }
        
        device_id = simulator.create_simulated_device(device_config)
        
        # Test individual device status
        status = simulator.get_device_status(device_id)
        
        assert status is not None
        assert status['device_id'] == device_id
        assert status['device_name'] == "Status_Test_Device"
        assert status['device_type'] == "cloud"
        assert status['compute_power'] == 5.0
        
        # Test non-existent device
        invalid_status = simulator.get_device_status("invalid_id")
        assert invalid_status is None
        
        # Test all devices status
        all_statuses = simulator.get_all_devices_status()
        assert len(all_statuses) == 1
        assert all_statuses[0]['device_id'] == device_id

class TestPySyftIntegration:
    """Unit tests for PySyft integration"""
    
    def test_initialization(self):
        """Test PySyft integration initialization"""
        integration = PySyftIntegration()
        
        # Should initialize without error regardless of PySyft availability
        assert integration.workers == {}
        assert integration.virtual_workers == {}
    
    def test_virtual_worker_creation(self):
        """Test virtual worker creation"""
        integration = PySyftIntegration()
        
        worker_id = "test_worker"
        worker = integration.create_virtual_worker(worker_id)
        
        if integration.hook:  # PySyft available
            assert worker is not None
            assert worker_id in integration.virtual_workers
        else:  # PySyft not available
            assert worker is None
    
    def test_federated_averaging(self):
        """Test federated averaging functionality"""
        integration = PySyftIntegration()
        
        # Test with mock models
        mock_models = [Mock() for _ in range(3)]
        result = integration.federated_averaging(mock_models)
        
        # Should handle gracefully whether PySyft is available or not
        if not integration.hook:
            assert result is None
    
    @patch('scrollintel.engines.federated_engine.HAS_SYFT', True)
    @patch('scrollintel.engines.federated_engine.sy')
    def test_with_mocked_syft(self, mock_syft):
        """Test PySyft integration with mocked PySyft"""
        mock_hook = Mock()
        mock_syft.TorchHook.return_value = mock_hook
        
        integration = PySyftIntegration()
        assert integration.hook == mock_hook
        
        # Test worker creation with mock
        mock_worker = Mock()
        mock_syft.VirtualWorker.return_value = mock_worker
        
        worker_id = "mock_worker"
        worker = integration.create_virtual_worker(worker_id)
        
        assert worker == mock_worker
        assert worker_id in integration.virtual_workers

class TestTensorFlowFederatedIntegration:
    """Unit tests for TensorFlow Federated integration"""
    
    def test_initialization(self):
        """Test TFF integration initialization"""
        integration = TensorFlowFederatedIntegration()
        
        assert integration.client_data is None
        assert integration.model_fn is None
        assert integration.federated_process is None
    
    def test_keras_model_creation(self):
        """Test Keras model creation for federated learning"""
        integration = TensorFlowFederatedIntegration()
        
        input_shape = (784,)
        num_classes = 10
        
        model_fn = integration.create_keras_model(input_shape, num_classes)
        
        # Should return None if TFF not available, function if available
        if model_fn is not None:
            assert callable(model_fn)
            assert integration.model_fn == model_fn
    
    def test_federated_data_creation(self):
        """Test federated data creation"""
        integration = TensorFlowFederatedIntegration()
        
        client_datasets = {
            "client_1": (np.random.rand(100, 784).astype(np.float32), 
                        np.random.randint(0, 10, 100).astype(np.int32)),
            "client_2": (np.random.rand(150, 784).astype(np.float32), 
                        np.random.randint(0, 10, 150).astype(np.int32))
        }
        
        federated_data = integration.create_federated_data(client_datasets)
        
        # Should handle gracefully whether TFF is available or not
        if federated_data is not None:
            assert integration.client_data == federated_data

class TestFederatedEngineCore:
    """Unit tests for core FederatedEngine functionality"""
    
    @pytest.fixture
    def federated_engine(self):
        """Create federated engine for testing"""
        return FederatedEngine()
    
    def test_initialization(self, federated_engine):
        """Test federated engine initialization"""
        engine = federated_engine
        
        assert engine.status == FederatedEngineStatus.READY
        assert engine.tasks == {}
        assert engine.active_task is None
        assert isinstance(engine.privacy_engine, DifferentialPrivacyEngine)
        assert isinstance(engine.secure_aggregation, SecureAggregationProtocol)
        assert isinstance(engine.device_simulator, EdgeDeviceSimulator)
        assert 'min_clients' in engine.config
        assert 'max_clients' in engine.config
    
    @pytest.mark.asyncio
    async def test_task_creation(self, federated_engine):
        """Test federated task creation"""
        engine = federated_engine
        
        task_config = {
            "task_name": "Unit_Test_Task",
            "model_architecture": {
                "input_size": 10,
                "output_size": 2
            },
            "participating_devices": ["device_1", "device_2"]
        }
        
        task_id = await engine.create_federated_task(task_config)
        
        assert task_id is not None
        assert task_id in engine.tasks
        
        task = engine.tasks[task_id]
        assert task.task_name == "Unit_Test_Task"
        assert task.status == FederatedEngineStatus.READY
        assert task.rounds_completed == 0
    
    @pytest.mark.asyncio
    async def test_device_management(self, federated_engine):
        """Test device management functionality"""
        engine = federated_engine
        
        device_config = {
            "device_name": "Unit_Test_Device",
            "device_type": EdgeDeviceType.DESKTOP,
            "data_size": 500
        }
        
        # Add device
        device_id = await engine.add_edge_device(device_config)
        assert device_id is not None
        assert device_id in engine.device_simulator.devices
        
        # Remove device
        removed = await engine.remove_edge_device(device_id)
        assert removed is True
        assert device_id not in engine.device_simulator.devices
        
        # Try to remove non-existent device
        not_removed = await engine.remove_edge_device("invalid_id")
        assert not_removed is False
    
    @pytest.mark.asyncio
    async def test_federation_status(self, federated_engine):
        """Test federation status reporting"""
        engine = federated_engine
        
        # Add some devices and tasks
        device_config = {"device_name": "Status_Device", "device_type": EdgeDeviceType.MOBILE}
        device_id = await engine.add_edge_device(device_config)
        
        task_config = {
            "task_name": "Status_Task",
            "model_architecture": {"input_size": 5, "output_size": 1},
            "participating_devices": [device_id]
        }
        task_id = await engine.create_federated_task(task_config)
        
        # Get status
        status = await engine.get_federation_status()
        
        assert status['engine_status'] == 'ready'
        assert status['total_tasks'] == 1
        assert status['total_devices'] == 1
        assert status['online_devices'] >= 0
        assert 'privacy_budget_remaining' in status
        assert 'config' in status
    
    def test_pytorch_model_creation(self, federated_engine):
        """Test PyTorch model creation"""
        engine = federated_engine
        
        architecture = {
            "input_size": 784,
            "hidden_size": 128,
            "output_size": 10
        }
        
        model = engine._create_pytorch_model(architecture)
        
        # Should create model if PyTorch available, None otherwise
        from scrollintel.engines.federated_engine import HAS_TORCH
        if HAS_TORCH:
            assert model is not None
            assert hasattr(model, 'forward')
        else:
            assert model is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])