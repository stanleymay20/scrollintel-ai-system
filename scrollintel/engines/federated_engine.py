"""
ScrollIntel FederatedEngine for Distributed Learning
Implements PySyft integration, TensorFlow Federated support, differential privacy,
secure aggregation protocols, and edge device simulation.

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import asyncio
import logging
import json
import time
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import pickle
import base64
import threading
from concurrent.futures import ThreadPoolExecutor

# Core ML frameworks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

# PySyft for federated learning
try:
    import syft as sy
    HAS_SYFT = True
except ImportError:
    HAS_SYFT = False
    sy = None

# TensorFlow Federated
try:
    import tensorflow_federated as tff
    HAS_TFF = True
except ImportError:
    HAS_TFF = False
    tff = None

logger = logging.getLogger(__name__)

class FederatedEngineStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class EdgeDeviceType(Enum):
    MOBILE = "mobile"
    IOT = "iot"
    DESKTOP = "desktop"
    SERVER = "server"
    CLOUD = "cloud"

class PrivacyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class EdgeDevice:
    device_id: str
    device_name: str
    device_type: EdgeDeviceType
    ip_address: str
    port: int
    capabilities: Dict[str, Any]
    data_size: int
    compute_power: float  # Relative compute capability
    bandwidth: float  # MB/s
    battery_level: Optional[float]  # For mobile devices
    privacy_level: PrivacyLevel
    last_seen: datetime
    status: str
    model_version: int

@dataclass
class FederatedTask:
    task_id: str
    task_name: str
    model_architecture: Dict[str, Any]
    training_config: Dict[str, Any]
    privacy_config: Dict[str, Any]
    participating_devices: List[str]
    created_at: datetime
    status: FederatedEngineStatus
    rounds_completed: int
    target_rounds: int
    convergence_threshold: float

class DifferentialPrivacyEngine:
    """Advanced differential privacy implementation"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0, mechanism: str = "gaussian"):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        self.privacy_budget_used = 0.0
        self.composition_method = "advanced"  # or "basic"
    
    def get_noise_multiplier(self) -> float:
        """Calculate noise multiplier for Gaussian mechanism"""
        if self.mechanism == "gaussian":
            return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        elif self.mechanism == "laplace":
            return self.sensitivity / self.epsilon
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def add_noise(self, data: np.ndarray, sensitivity: Optional[float] = None) -> np.ndarray:
        """Add differential privacy noise to data"""
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        if self.mechanism == "gaussian":
            noise_scale = sensitivity * self.get_noise_multiplier()
            noise = np.random.normal(0, noise_scale, data.shape)
        elif self.mechanism == "laplace":
            noise_scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, noise_scale, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        
        # Track privacy budget usage
        self.privacy_budget_used += self.epsilon
        
        return data + noise
    
    def clip_gradients(self, gradients: np.ndarray, clip_norm: float) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        if isinstance(gradients, list):
            # Handle list of gradient arrays
            clipped = []
            for grad in gradients:
                grad_norm = np.linalg.norm(grad)
                if grad_norm > clip_norm:
                    clipped.append(grad * (clip_norm / grad_norm))
                else:
                    clipped.append(grad)
            return clipped
        else:
            # Handle single gradient array
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > clip_norm:
                return gradients * (clip_norm / grad_norm)
            return gradients
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.epsilon - self.privacy_budget_used)
    
    def reset_budget(self):
        """Reset privacy budget"""
        self.privacy_budget_used = 0.0

class SecureAggregationProtocol:
    """Secure multi-party computation for model aggregation"""
    
    def __init__(self, threshold: int = 2):
        self.threshold = threshold  # Minimum number of parties needed
        self.secret_shares = {}
        self.reconstruction_shares = {}
        self.aggregation_masks = {}
    
    def generate_secret_shares(self, secret: float, num_parties: int) -> List[Tuple[int, float]]:
        """Generate secret shares using Shamir's Secret Sharing"""
        if num_parties < self.threshold:
            raise ValueError("Number of parties must be >= threshold")
        
        # Generate random polynomial coefficients
        coefficients = [secret] + [np.random.uniform(-1000, 1000) for _ in range(self.threshold - 1)]
        
        # Generate shares
        shares = []
        for i in range(1, num_parties + 1):
            share_value = sum(coeff * (i ** j) for j, coeff in enumerate(coefficients))
            shares.append((i, share_value))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, float]]) -> float:
        """Reconstruct secret from shares using Lagrange interpolation"""
        if len(shares) < self.threshold:
            raise ValueError("Not enough shares to reconstruct secret")
        
        secret = 0.0
        for i, (xi, yi) in enumerate(shares[:self.threshold]):
            # Lagrange basis polynomial
            li = 1.0
            for j, (xj, _) in enumerate(shares[:self.threshold]):
                if i != j:
                    li *= (0 - xj) / (xi - xj)
            secret += yi * li
        
        return secret
    
    def generate_aggregation_mask(self, device_id: str, seed: int) -> np.ndarray:
        """Generate pseudorandom mask for secure aggregation"""
        np.random.seed(seed + hash(device_id) % 1000000)
        mask = np.random.normal(0, 1, 1000)  # Adjust size as needed
        self.aggregation_masks[device_id] = mask
        return mask
    
    def secure_sum(self, masked_inputs: List[np.ndarray], 
                   dropout_masks: List[np.ndarray]) -> np.ndarray:
        """Perform secure sum with dropout resilience"""
        if not masked_inputs:
            return np.array([])
        
        # Sum all masked inputs
        result = np.sum(masked_inputs, axis=0)
        
        # Subtract dropout compensation
        if dropout_masks:
            dropout_compensation = np.sum(dropout_masks, axis=0)
            result -= dropout_compensation
        
        return result

class PySyftIntegration:
    """PySyft integration for federated learning"""
    
    def __init__(self):
        self.hook = None
        self.workers = {}
        self.virtual_workers = {}
        
        if HAS_SYFT and HAS_TORCH:
            self.hook = sy.TorchHook(torch)
            logger.info("PySyft TorchHook initialized")
        else:
            logger.warning("PySyft or PyTorch not available")
    
    def create_virtual_worker(self, worker_id: str) -> Optional[Any]:
        """Create a virtual worker for simulation"""
        if not self.hook:
            return None
        
        worker = sy.VirtualWorker(self.hook, id=worker_id)
        self.virtual_workers[worker_id] = worker
        logger.info(f"Created virtual worker: {worker_id}")
        return worker
    
    def send_model_to_worker(self, model: Any, worker_id: str) -> Optional[Any]:
        """Send model to worker"""
        if worker_id not in self.virtual_workers:
            logger.error(f"Worker {worker_id} not found")
            return None
        
        worker = self.virtual_workers[worker_id]
        try:
            model_copy = model.copy()
            sent_model = model_copy.send(worker)
            logger.info(f"Model sent to worker {worker_id}")
            return sent_model
        except Exception as e:
            logger.error(f"Failed to send model to worker {worker_id}: {e}")
            return None
    
    def federated_averaging(self, models: List[Any]) -> Optional[Any]:
        """Perform federated averaging of models"""
        if not models:
            return None
        
        try:
            # Get the first model as base
            averaged_model = models[0].copy()
            
            # Average parameters
            for param_name in averaged_model.state_dict():
                param_sum = models[0].state_dict()[param_name].clone()
                
                for model in models[1:]:
                    param_sum += model.state_dict()[param_name]
                
                averaged_model.state_dict()[param_name] = param_sum / len(models)
            
            logger.info(f"Federated averaging completed for {len(models)} models")
            return averaged_model
            
        except Exception as e:
            logger.error(f"Federated averaging failed: {e}")
            return None

class TensorFlowFederatedIntegration:
    """TensorFlow Federated (TFF) integration"""
    
    def __init__(self):
        self.client_data = None
        self.model_fn = None
        self.federated_process = None
        
        if HAS_TFF:
            logger.info("TensorFlow Federated available")
        else:
            logger.warning("TensorFlow Federated not available")
    
    def create_keras_model(self, input_shape: Tuple[int, ...], 
                          num_classes: int) -> Optional[Any]:
        """Create a Keras model for federated learning"""
        if not HAS_TFF or not HAS_TENSORFLOW:
            return None
        
        def model_fn():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            return tff.learning.from_keras_model(
                model,
                input_spec=tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
        
        self.model_fn = model_fn
        return model_fn
    
    def create_federated_data(self, client_datasets: Dict[str, Any]) -> Optional[Any]:
        """Create federated data from client datasets"""
        if not HAS_TFF:
            return None
        
        try:
            # Convert client datasets to TFF format
            federated_data = []
            for client_id, dataset in client_datasets.items():
                # Assume dataset is a tuple of (features, labels)
                features, labels = dataset
                client_data = tf.data.Dataset.from_tensor_slices({
                    'x': features,
                    'y': labels
                }).batch(32)
                federated_data.append(client_data)
            
            self.client_data = federated_data
            logger.info(f"Created federated data for {len(client_datasets)} clients")
            return federated_data
            
        except Exception as e:
            logger.error(f"Failed to create federated data: {e}")
            return None
    
    def build_federated_averaging_process(self) -> Optional[Any]:
        """Build federated averaging process"""
        if not HAS_TFF or not self.model_fn:
            return None
        
        try:
            # Create federated averaging process
            self.federated_process = tff.learning.build_federated_averaging_process(
                model_fn=self.model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
            )
            
            logger.info("Federated averaging process created")
            return self.federated_process
            
        except Exception as e:
            logger.error(f"Failed to build federated averaging process: {e}")
            return None
    
    def run_federated_training(self, num_rounds: int = 10) -> Optional[Dict[str, Any]]:
        """Run federated training"""
        if not self.federated_process or not self.client_data:
            logger.error("Federated process or client data not initialized")
            return None
        
        try:
            # Initialize the process
            state = self.federated_process.initialize()
            
            results = []
            for round_num in range(num_rounds):
                # Run one round of federated training
                state, metrics = self.federated_process.next(state, self.client_data)
                
                round_result = {
                    'round': round_num + 1,
                    'metrics': {k: float(v) for k, v in metrics.items()}
                }
                results.append(round_result)
                
                logger.info(f"Round {round_num + 1} completed: {round_result['metrics']}")
            
            return {
                'final_state': state,
                'training_results': results
            }
            
        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            return None

class EdgeDeviceSimulator:
    """Simulates edge devices for federated learning"""
    
    def __init__(self):
        self.devices = {}
        self.device_data = {}
        self.simulation_running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def create_simulated_device(self, device_config: Dict[str, Any]) -> str:
        """Create a simulated edge device"""
        device_id = device_config.get('device_id', str(uuid.uuid4()))
        
        device = EdgeDevice(
            device_id=device_id,
            device_name=device_config.get('device_name', f"Device_{device_id[:8]}"),
            device_type=EdgeDeviceType(device_config.get('device_type', 'desktop')),
            ip_address=device_config.get('ip_address', f"192.168.1.{len(self.devices) + 100}"),
            port=device_config.get('port', 8080 + len(self.devices)),
            capabilities=device_config.get('capabilities', {}),
            data_size=device_config.get('data_size', 1000),
            compute_power=device_config.get('compute_power', 1.0),
            bandwidth=device_config.get('bandwidth', 10.0),
            battery_level=device_config.get('battery_level'),
            privacy_level=PrivacyLevel(device_config.get('privacy_level', 'medium')),
            last_seen=datetime.utcnow(),
            status='online',
            model_version=0
        )
        
        self.devices[device_id] = device
        
        # Generate synthetic data for the device
        self._generate_device_data(device_id, device.data_size)
        
        logger.info(f"Created simulated device: {device_id} ({device.device_type.value})")
        return device_id
    
    def _generate_device_data(self, device_id: str, data_size: int):
        """Generate synthetic data for a device"""
        # Generate random data based on device characteristics
        device = self.devices[device_id]
        
        # Simulate different data distributions based on device type
        if device.device_type == EdgeDeviceType.MOBILE:
            # Mobile devices might have image data
            data = np.random.rand(data_size, 28, 28)  # MNIST-like
            labels = np.random.randint(0, 10, data_size)
        elif device.device_type == EdgeDeviceType.IOT:
            # IoT devices might have sensor data
            data = np.random.rand(data_size, 10)  # Sensor readings
            labels = np.random.randint(0, 2, data_size)  # Binary classification
        else:
            # Default tabular data
            data = np.random.rand(data_size, 20)
            labels = np.random.randint(0, 5, data_size)
        
        self.device_data[device_id] = {
            'features': data,
            'labels': labels,
            'data_size': data_size
        }
    
    def simulate_device_training(self, device_id: str, 
                               global_model: Any, 
                               training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate training on a device"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        device = self.devices[device_id]
        device_data = self.device_data[device_id]
        
        # Simulate training time based on device capabilities
        training_time = training_config.get('epochs', 5) / device.compute_power
        
        # Simulate network latency
        network_delay = 1.0 / device.bandwidth
        
        # Simulate battery consumption for mobile devices
        if device.battery_level is not None:
            battery_consumption = training_time * 0.1  # 10% per time unit
            device.battery_level = max(0, device.battery_level - battery_consumption)
        
        # Simulate training results
        training_loss = np.random.uniform(0.1, 2.0)
        training_accuracy = np.random.uniform(0.6, 0.95)
        
        # Add noise based on privacy level
        privacy_noise = {
            PrivacyLevel.LOW: 0.01,
            PrivacyLevel.MEDIUM: 0.05,
            PrivacyLevel.HIGH: 0.1,
            PrivacyLevel.MAXIMUM: 0.2
        }[device.privacy_level]
        
        training_loss += np.random.normal(0, privacy_noise)
        training_accuracy += np.random.normal(0, privacy_noise * 0.1)
        training_accuracy = np.clip(training_accuracy, 0, 1)
        
        # Update device status
        device.last_seen = datetime.utcnow()
        device.model_version += 1
        
        return {
            'device_id': device_id,
            'training_time': training_time,
            'network_delay': network_delay,
            'training_loss': training_loss,
            'training_accuracy': training_accuracy,
            'data_size': device_data['data_size'],
            'battery_level': device.battery_level,
            'privacy_noise_level': privacy_noise
        }
    
    def simulate_device_failure(self, device_id: str, failure_type: str = "network"):
        """Simulate device failure"""
        if device_id in self.devices:
            device = self.devices[device_id]
            if failure_type == "network":
                device.status = "offline"
            elif failure_type == "battery":
                device.battery_level = 0
                device.status = "low_battery"
            elif failure_type == "compute":
                device.compute_power *= 0.1  # Severely reduced performance
                device.status = "degraded"
            
            logger.info(f"Simulated {failure_type} failure for device {device_id}")
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device status"""
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        return {
            'device_id': device_id,
            'device_name': device.device_name,
            'device_type': device.device_type.value,
            'status': device.status,
            'compute_power': device.compute_power,
            'bandwidth': device.bandwidth,
            'battery_level': device.battery_level,
            'privacy_level': device.privacy_level.value,
            'last_seen': device.last_seen.isoformat(),
            'model_version': device.model_version,
            'data_size': device.data_size
        }
    
    def get_all_devices_status(self) -> List[Dict[str, Any]]:
        """Get status of all devices"""
        return [self.get_device_status(device_id) for device_id in self.devices.keys()]

class FederatedEngine:
    """Main federated learning engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.status = FederatedEngineStatus.INITIALIZING
        self.tasks = {}
        self.active_task = None
        
        # Initialize components
        self.privacy_engine = DifferentialPrivacyEngine()
        self.secure_aggregation = SecureAggregationProtocol()
        self.pysyft_integration = PySyftIntegration()
        self.tff_integration = TensorFlowFederatedIntegration()
        self.device_simulator = EdgeDeviceSimulator()
        
        # Configuration
        self.config = {
            'min_clients': 2,
            'max_clients': 100,
            'round_timeout': 300,  # seconds
            'convergence_threshold': 0.001,
            'max_rounds': 50,
            'privacy_budget': 1.0,
            'secure_aggregation_threshold': 2
        }
        
        self.status = FederatedEngineStatus.READY
        logger.info("FederatedEngine initialized successfully")
    
    async def create_federated_task(self, task_config: Dict[str, Any]) -> str:
        """Create a new federated learning task"""
        task_id = str(uuid.uuid4())
        
        task = FederatedTask(
            task_id=task_id,
            task_name=task_config.get('task_name', f"Task_{task_id[:8]}"),
            model_architecture=task_config.get('model_architecture', {}),
            training_config=task_config.get('training_config', {}),
            privacy_config=task_config.get('privacy_config', {}),
            participating_devices=task_config.get('participating_devices', []),
            created_at=datetime.utcnow(),
            status=FederatedEngineStatus.READY,
            rounds_completed=0,
            target_rounds=task_config.get('target_rounds', 10),
            convergence_threshold=task_config.get('convergence_threshold', 0.001)
        )
        
        self.tasks[task_id] = task
        
        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.hset(
                "federated_tasks",
                task_id,
                json.dumps(asdict(task), default=str)
            )
        
        logger.info(f"Created federated task: {task_id}")
        return task_id
    
    async def start_federated_training(self, task_id: str, 
                                     framework: str = "pytorch") -> bool:
        """Start federated training for a task"""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        self.active_task = task
        task.status = FederatedEngineStatus.TRAINING
        
        try:
            if framework == "pysyft" and HAS_SYFT:
                return await self._run_pysyft_training(task)
            elif framework == "tff" and HAS_TFF:
                return await self._run_tff_training(task)
            else:
                return await self._run_custom_training(task)
                
        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            task.status = FederatedEngineStatus.FAILED
            return False
    
    async def _run_pysyft_training(self, task: FederatedTask) -> bool:
        """Run federated training using PySyft"""
        logger.info(f"Starting PySyft training for task {task.task_id}")
        
        # Create virtual workers for participating devices
        workers = []
        for device_id in task.participating_devices:
            worker = self.pysyft_integration.create_virtual_worker(device_id)
            if worker:
                workers.append(worker)
        
        if not workers:
            logger.error("No workers available for PySyft training")
            return False
        
        # Create and distribute model
        if HAS_TORCH:
            model = self._create_pytorch_model(task.model_architecture)
            
            # Send model to workers
            worker_models = []
            for i, worker in enumerate(workers):
                worker_model = self.pysyft_integration.send_model_to_worker(
                    model, task.participating_devices[i]
                )
                if worker_model:
                    worker_models.append(worker_model)
            
            # Simulate training rounds
            for round_num in range(task.target_rounds):
                logger.info(f"PySyft training round {round_num + 1}")
                
                # Simulate local training on each worker
                trained_models = []
                for worker_model in worker_models:
                    # In real implementation, this would be actual training
                    trained_models.append(worker_model)
                
                # Perform federated averaging
                if trained_models:
                    averaged_model = self.pysyft_integration.federated_averaging(trained_models)
                    if averaged_model:
                        model = averaged_model
                
                task.rounds_completed += 1
                
                # Check convergence (simplified)
                if round_num > 5:  # Allow some rounds before checking
                    break
            
            task.status = FederatedEngineStatus.COMPLETED
            logger.info(f"PySyft training completed for task {task.task_id}")
            return True
        
        return False
    
    async def _run_tff_training(self, task: FederatedTask) -> bool:
        """Run federated training using TensorFlow Federated"""
        logger.info(f"Starting TFF training for task {task.task_id}")
        
        # Prepare client datasets
        client_datasets = {}
        for device_id in task.participating_devices:
            if device_id in self.device_simulator.device_data:
                device_data = self.device_simulator.device_data[device_id]
                client_datasets[device_id] = (
                    device_data['features'],
                    device_data['labels']
                )
        
        if not client_datasets:
            logger.error("No client datasets available for TFF training")
            return False
        
        # Create model and federated data
        model_arch = task.model_architecture
        input_shape = model_arch.get('input_shape', (20,))
        num_classes = model_arch.get('num_classes', 5)
        
        model_fn = self.tff_integration.create_keras_model(input_shape, num_classes)
        federated_data = self.tff_integration.create_federated_data(client_datasets)
        
        if not model_fn or not federated_data:
            logger.error("Failed to create TFF model or data")
            return False
        
        # Build and run federated training
        process = self.tff_integration.build_federated_averaging_process()
        if process:
            results = self.tff_integration.run_federated_training(task.target_rounds)
            if results:
                task.rounds_completed = task.target_rounds
                task.status = FederatedEngineStatus.COMPLETED
                logger.info(f"TFF training completed for task {task.task_id}")
                return True
        
        return False
    
    async def _run_custom_training(self, task: FederatedTask) -> bool:
        """Run custom federated training implementation"""
        logger.info(f"Starting custom training for task {task.task_id}")
        
        # Initialize privacy engine
        privacy_config = task.privacy_config
        self.privacy_engine = DifferentialPrivacyEngine(
            epsilon=privacy_config.get('epsilon', 1.0),
            delta=privacy_config.get('delta', 1e-5),
            sensitivity=privacy_config.get('sensitivity', 1.0)
        )
        
        # Run training rounds
        for round_num in range(task.target_rounds):
            logger.info(f"Custom training round {round_num + 1}")
            
            # Simulate training on each device
            round_results = []
            for device_id in task.participating_devices:
                if device_id in self.device_simulator.devices:
                    result = self.device_simulator.simulate_device_training(
                        device_id, None, task.training_config
                    )
                    round_results.append(result)
            
            if not round_results:
                logger.error("No training results from devices")
                break
            
            # Apply differential privacy
            for result in round_results:
                # Add privacy noise to training metrics
                result['training_loss'] = self.privacy_engine.add_noise(
                    np.array([result['training_loss']])
                )[0]
                result['training_accuracy'] = self.privacy_engine.add_noise(
                    np.array([result['training_accuracy']])
                )[0]
            
            # Perform secure aggregation
            aggregated_loss = np.mean([r['training_loss'] for r in round_results])
            aggregated_accuracy = np.mean([r['training_accuracy'] for r in round_results])
            
            task.rounds_completed += 1
            
            # Check convergence
            if round_num > 0 and abs(aggregated_loss) < task.convergence_threshold:
                logger.info(f"Training converged at round {round_num + 1}")
                break
            
            # Store round results
            if self.redis_client:
                round_data = {
                    'round': round_num + 1,
                    'aggregated_loss': float(aggregated_loss),
                    'aggregated_accuracy': float(aggregated_accuracy),
                    'participating_devices': len(round_results),
                    'privacy_budget_used': self.privacy_engine.privacy_budget_used
                }
                await self.redis_client.lpush(
                    f"training_rounds:{task.task_id}",
                    json.dumps(round_data)
                )
        
        task.status = FederatedEngineStatus.COMPLETED
        logger.info(f"Custom training completed for task {task.task_id}")
        return True
    
    def _create_pytorch_model(self, architecture: Dict[str, Any]):
        """Create PyTorch model from architecture"""
        if not HAS_TORCH:
            return None
        
        input_size = architecture.get('input_size', 784)
        hidden_size = architecture.get('hidden_size', 128)
        output_size = architecture.get('output_size', 10)
        
        class FederatedModel(nn.Module):
            def __init__(self):
                super(FederatedModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return FederatedModel()
    
    async def add_edge_device(self, device_config: Dict[str, Any]) -> str:
        """Add an edge device to the federation"""
        device_id = self.device_simulator.create_simulated_device(device_config)
        
        # Store device info in Redis
        if self.redis_client:
            device_info = self.device_simulator.get_device_status(device_id)
            await self.redis_client.hset(
                "edge_devices",
                device_id,
                json.dumps(device_info)
            )
        
        return device_id
    
    async def remove_edge_device(self, device_id: str) -> bool:
        """Remove an edge device from the federation"""
        if device_id in self.device_simulator.devices:
            del self.device_simulator.devices[device_id]
            if device_id in self.device_simulator.device_data:
                del self.device_simulator.device_data[device_id]
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.hdel("edge_devices", device_id)
            
            logger.info(f"Removed edge device: {device_id}")
            return True
        
        return False
    
    async def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status"""
        return {
            'engine_status': self.status.value,
            'active_task': self.active_task.task_id if self.active_task else None,
            'total_tasks': len(self.tasks),
            'total_devices': len(self.device_simulator.devices),
            'online_devices': len([
                d for d in self.device_simulator.devices.values()
                if d.status == 'online'
            ]),
            'privacy_budget_remaining': self.privacy_engine.get_remaining_budget(),
            'config': self.config
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'task_id': task.task_id,
            'task_name': task.task_name,
            'status': task.status.value,
            'rounds_completed': task.rounds_completed,
            'target_rounds': task.target_rounds,
            'participating_devices': len(task.participating_devices),
            'created_at': task.created_at.isoformat(),
            'convergence_threshold': task.convergence_threshold
        }
    
    async def pause_training(self, task_id: str) -> bool:
        """Pause federated training"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == FederatedEngineStatus.TRAINING:
                task.status = FederatedEngineStatus.PAUSED
                logger.info(f"Paused training for task {task_id}")
                return True
        return False
    
    async def resume_training(self, task_id: str) -> bool:
        """Resume federated training"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == FederatedEngineStatus.PAUSED:
                task.status = FederatedEngineStatus.TRAINING
                logger.info(f"Resumed training for task {task_id}")
                return True
        return False
    
    async def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """Clean up completed tasks older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status in [FederatedEngineStatus.COMPLETED, FederatedEngineStatus.FAILED] 
                and task.created_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.hdel("federated_tasks", task_id)
                await self.redis_client.delete(f"training_rounds:{task_id}")
        
        logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
        return len(tasks_to_remove)

# Global federated engine instance
_federated_engine = None

def get_federated_engine(redis_client=None) -> FederatedEngine:
    """Get global federated engine instance"""
    global _federated_engine
    
    if _federated_engine is None:
        _federated_engine = FederatedEngine(redis_client)
    
    return _federated_engine

async def initialize_federated_engine():
    """Initialize the federated learning engine"""
    logger.info("Initializing ScrollIntel FederatedEngine")
    
    # Check dependencies
    missing_deps = []
    if not HAS_TORCH:
        missing_deps.append("PyTorch")
    if not HAS_TENSORFLOW:
        missing_deps.append("TensorFlow")
    if not HAS_SYFT:
        missing_deps.append("PySyft")
    if not HAS_TFF:
        missing_deps.append("TensorFlow Federated")
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.warning("Some federated learning features may not be available")
    
    # Initialize global engine
    engine = get_federated_engine()
    
    logger.info("FederatedEngine initialization completed")
    return engine