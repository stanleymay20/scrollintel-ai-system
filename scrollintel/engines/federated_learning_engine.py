"""
ScrollIntel Federated Learning Engine
Implements privacy-preserving distributed machine learning capabilities.
"""

import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import pickle
import base64

# Optional imports for federated learning
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

logger = logging.getLogger(__name__)

class FederatedLearningStatus(Enum):
    INITIALIZING = "initializing"
    WAITING_FOR_CLIENTS = "waiting_for_clients"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"

class ClientStatus(Enum):
    CONNECTED = "connected"
    TRAINING = "training"
    COMPLETED = "completed"
    DISCONNECTED = "disconnected"
    FAILED = "failed"

@dataclass
class FederatedClient:
    client_id: str
    client_name: str
    ip_address: str
    status: ClientStatus
    data_size: int
    model_version: int
    last_update: datetime
    performance_metrics: Dict[str, float]
    privacy_budget: float
    capabilities: Dict[str, Any]

@dataclass
class ModelUpdate:
    client_id: str
    model_weights: bytes  # Serialized model weights
    gradient_updates: bytes  # Serialized gradients
    training_metrics: Dict[str, float]
    data_size: int
    privacy_noise: float
    timestamp: datetime
    signature: str  # For integrity verification

@dataclass
class FederatedRound:
    round_id: str
    round_number: int
    participating_clients: List[str]
    global_model_weights: bytes
    aggregated_metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime]
    status: FederatedLearningStatus

class DifferentialPrivacyManager:
    """Manages differential privacy for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Privacy parameter
        self.noise_multiplier = self._calculate_noise_multiplier()
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier based on privacy parameters"""
        # Simplified calculation - in production use more sophisticated methods
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise_to_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Add differential privacy noise to gradients"""
        if not isinstance(gradients, np.ndarray):
            gradients = np.array(gradients)
        
        # Calculate L2 sensitivity (simplified)
        l2_sensitivity = 1.0  # This should be calculated based on the model
        
        # Add Gaussian noise
        noise_scale = l2_sensitivity * self.noise_multiplier
        noise = np.random.normal(0, noise_scale, gradients.shape)
        
        return gradients + noise
    
    def clip_gradients(self, gradients: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        if not isinstance(gradients, np.ndarray):
            gradients = np.array(gradients)
        
        # Calculate L2 norm
        l2_norm = np.linalg.norm(gradients)
        
        # Clip if necessary
        if l2_norm > clip_norm:
            gradients = gradients * (clip_norm / l2_norm)
        
        return gradients
    
    def consume_privacy_budget(self, amount: float) -> bool:
        """Consume privacy budget"""
        if self.epsilon >= amount:
            self.epsilon -= amount
            return True
        return False

class SecureAggregation:
    """Implements secure aggregation for federated learning"""
    
    def __init__(self):
        self.aggregation_keys = {}
        self.client_shares = {}
    
    def generate_aggregation_keys(self, clients: List[str]) -> Dict[str, str]:
        """Generate keys for secure aggregation"""
        keys = {}
        for client_id in clients:
            # In production, use proper cryptographic key generation
            key = hashlib.sha256(f"{client_id}_{time.time()}".encode()).hexdigest()
            keys[client_id] = key
            self.aggregation_keys[client_id] = key
        
        return keys
    
    def create_secret_shares(self, value: float, num_shares: int) -> List[float]:
        """Create secret shares for secure aggregation"""
        # Simplified secret sharing - in production use Shamir's Secret Sharing
        shares = []
        total = 0
        
        for i in range(num_shares - 1):
            share = np.random.uniform(-1, 1)
            shares.append(share)
            total += share
        
        # Last share ensures sum equals original value
        shares.append(value - total)
        
        return shares
    
    def aggregate_secure_updates(
        self, 
        encrypted_updates: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Aggregate encrypted model updates securely"""
        if not encrypted_updates:
            return np.array([])
        
        # Simplified secure aggregation
        # In production, implement proper secure multi-party computation
        
        aggregated_weights = None
        total_weight = 0
        
        for update in encrypted_updates:
            client_id = update['client_id']
            weights = pickle.loads(base64.b64decode(update['weights']))
            data_size = update['data_size']
            
            if aggregated_weights is None:
                aggregated_weights = np.zeros_like(weights)
            
            # Weighted aggregation
            aggregated_weights += weights * data_size
            total_weight += data_size
        
        if total_weight > 0:
            aggregated_weights /= total_weight
        
        return aggregated_weights

class FederatedModelManager:
    """Manages federated learning models"""
    
    def __init__(self):
        self.global_model = None
        self.model_architecture = None
        self.model_version = 0
    
    def initialize_global_model(self, architecture: Dict[str, Any]) -> bool:
        """Initialize global model with specified architecture"""
        try:
            self.model_architecture = architecture
            
            if HAS_TORCH and architecture.get('framework') == 'pytorch':
                self.global_model = self._create_pytorch_model(architecture)
            elif HAS_TENSORFLOW and architecture.get('framework') == 'tensorflow':
                self.global_model = self._create_tensorflow_model(architecture)
            else:
                # Fallback to simple linear model
                self.global_model = self._create_simple_model(architecture)
            
            self.model_version = 1
            logger.info(f"Global model initialized with architecture: {architecture}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize global model: {e}")
            return False
    
    def _create_pytorch_model(self, architecture: Dict[str, Any]):
        """Create PyTorch model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        
        # Simple neural network example
        input_size = architecture.get('input_size', 784)
        hidden_size = architecture.get('hidden_size', 128)
        output_size = architecture.get('output_size', 10)
        
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return SimpleNN()
    
    def _create_tensorflow_model(self, architecture: Dict[str, Any]):
        """Create TensorFlow model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
        
        input_size = architecture.get('input_size', 784)
        hidden_size = architecture.get('hidden_size', 128)
        output_size = architecture.get('output_size', 10)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_simple_model(self, architecture: Dict[str, Any]):
        """Create simple model without deep learning frameworks"""
        # Simple linear model using numpy
        input_size = architecture.get('input_size', 784)
        output_size = architecture.get('output_size', 10)
        
        return {
            'weights': np.random.normal(0, 0.1, (input_size, output_size)),
            'bias': np.zeros(output_size),
            'architecture': architecture
        }
    
    def get_model_weights(self) -> bytes:
        """Get serialized model weights"""
        if self.global_model is None:
            return b''
        
        try:
            if HAS_TORCH and hasattr(self.global_model, 'state_dict'):
                # PyTorch model
                weights = self.global_model.state_dict()
                return pickle.dumps(weights)
            elif HAS_TENSORFLOW and hasattr(self.global_model, 'get_weights'):
                # TensorFlow model
                weights = self.global_model.get_weights()
                return pickle.dumps(weights)
            else:
                # Simple model
                return pickle.dumps(self.global_model)
        
        except Exception as e:
            logger.error(f"Failed to serialize model weights: {e}")
            return b''
    
    def update_model_weights(self, weights_data: bytes) -> bool:
        """Update global model with new weights"""
        try:
            weights = pickle.loads(weights_data)
            
            if HAS_TORCH and hasattr(self.global_model, 'load_state_dict'):
                # PyTorch model
                self.global_model.load_state_dict(weights)
            elif HAS_TENSORFLOW and hasattr(self.global_model, 'set_weights'):
                # TensorFlow model
                self.global_model.set_weights(weights)
            else:
                # Simple model
                self.global_model = weights
            
            self.model_version += 1
            logger.info(f"Global model updated to version {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
            return False

class FederatedLearningCoordinator:
    """Coordinates federated learning rounds"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.clients = {}
        self.current_round = None
        self.rounds_history = []
        self.model_manager = FederatedModelManager()
        self.privacy_manager = DifferentialPrivacyManager()
        self.secure_aggregation = SecureAggregation()
        self.min_clients = 2
        self.max_rounds = 100
        self.convergence_threshold = 0.001
    
    async def register_client(self, client_info: Dict[str, Any]) -> str:
        """Register a new federated learning client"""
        client_id = client_info.get('client_id') or f"client_{len(self.clients)}"
        
        client = FederatedClient(
            client_id=client_id,
            client_name=client_info.get('client_name', f"Client {client_id}"),
            ip_address=client_info.get('ip_address', ''),
            status=ClientStatus.CONNECTED,
            data_size=client_info.get('data_size', 0),
            model_version=0,
            last_update=datetime.utcnow(),
            performance_metrics={},
            privacy_budget=client_info.get('privacy_budget', 1.0),
            capabilities=client_info.get('capabilities', {})
        )
        
        self.clients[client_id] = client
        
        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.hset(
                "federated_clients",
                client_id,
                json.dumps(asdict(client), default=str)
            )
        
        logger.info(f"Registered federated client: {client_id}")
        return client_id
    
    async def start_federated_round(
        self, 
        model_architecture: Dict[str, Any]
    ) -> Optional[str]:
        """Start a new federated learning round"""
        if len(self.clients) < self.min_clients:
            logger.warning(f"Not enough clients to start round. Need {self.min_clients}, have {len(self.clients)}")
            return None
        
        # Initialize global model if not done
        if self.model_manager.global_model is None:
            if not self.model_manager.initialize_global_model(model_architecture):
                logger.error("Failed to initialize global model")
                return None
        
        # Create new round
        round_id = f"round_{int(time.time())}"
        participating_clients = list(self.clients.keys())
        
        self.current_round = FederatedRound(
            round_id=round_id,
            round_number=len(self.rounds_history) + 1,
            participating_clients=participating_clients,
            global_model_weights=self.model_manager.get_model_weights(),
            aggregated_metrics={},
            start_time=datetime.utcnow(),
            end_time=None,
            status=FederatedLearningStatus.TRAINING
        )
        
        # Generate secure aggregation keys
        aggregation_keys = self.secure_aggregation.generate_aggregation_keys(
            participating_clients
        )
        
        # Notify clients to start training
        for client_id in participating_clients:
            await self._notify_client_start_training(client_id, aggregation_keys[client_id])
        
        logger.info(f"Started federated round {round_id} with {len(participating_clients)} clients")
        return round_id
    
    async def receive_model_update(self, update: ModelUpdate) -> bool:
        """Receive model update from client"""
        if self.current_round is None:
            logger.warning("Received model update but no active round")
            return False
        
        client_id = update.client_id
        if client_id not in self.clients:
            logger.warning(f"Received update from unknown client: {client_id}")
            return False
        
        # Verify update integrity
        if not self._verify_update_signature(update):
            logger.warning(f"Invalid signature for update from client: {client_id}")
            return False
        
        # Apply differential privacy
        try:
            gradients = pickle.loads(update.gradient_updates)
            if isinstance(gradients, np.ndarray):
                # Clip and add noise to gradients
                clipped_gradients = self.privacy_manager.clip_gradients(gradients)
                noisy_gradients = self.privacy_manager.add_noise_to_gradients(clipped_gradients)
                update.gradient_updates = pickle.dumps(noisy_gradients)
        except Exception as e:
            logger.error(f"Failed to apply differential privacy: {e}")
        
        # Store update
        update_key = f"model_update:{self.current_round.round_id}:{client_id}"
        if self.redis_client:
            await self.redis_client.set(
                update_key,
                pickle.dumps(update),
                ex=3600  # Expire after 1 hour
            )
        
        # Update client status
        self.clients[client_id].status = ClientStatus.COMPLETED
        self.clients[client_id].last_update = datetime.utcnow()
        self.clients[client_id].performance_metrics = update.training_metrics
        
        logger.info(f"Received model update from client: {client_id}")
        
        # Check if all clients have submitted updates
        completed_clients = [
            c for c in self.current_round.participating_clients
            if self.clients[c].status == ClientStatus.COMPLETED
        ]
        
        if len(completed_clients) == len(self.current_round.participating_clients):
            await self._aggregate_model_updates()
        
        return True
    
    async def _aggregate_model_updates(self):
        """Aggregate model updates from all clients"""
        if self.current_round is None:
            return
        
        logger.info(f"Aggregating model updates for round {self.current_round.round_id}")
        
        # Collect all updates
        updates = []
        for client_id in self.current_round.participating_clients:
            update_key = f"model_update:{self.current_round.round_id}:{client_id}"
            if self.redis_client:
                update_data = await self.redis_client.get(update_key)
                if update_data:
                    update = pickle.loads(update_data)
                    updates.append({
                        'client_id': client_id,
                        'weights': base64.b64encode(update.model_weights).decode(),
                        'data_size': update.data_size,
                        'metrics': update.training_metrics
                    })
        
        if not updates:
            logger.error("No model updates found for aggregation")
            self.current_round.status = FederatedLearningStatus.FAILED
            return
        
        # Perform secure aggregation
        try:
            aggregated_weights = self.secure_aggregation.aggregate_secure_updates(updates)
            
            # Update global model
            aggregated_weights_bytes = pickle.dumps(aggregated_weights)
            if self.model_manager.update_model_weights(aggregated_weights_bytes):
                self.current_round.global_model_weights = aggregated_weights_bytes
                self.current_round.status = FederatedLearningStatus.COMPLETED
                self.current_round.end_time = datetime.utcnow()
                
                # Calculate aggregated metrics
                self.current_round.aggregated_metrics = self._calculate_aggregated_metrics(updates)
                
                # Add to history
                self.rounds_history.append(self.current_round)
                
                logger.info(f"Successfully aggregated round {self.current_round.round_id}")
                
                # Check for convergence
                if self._check_convergence():
                    logger.info("Federated learning has converged")
                    await self._finalize_federated_learning()
                else:
                    # Start next round if not converged and under max rounds
                    if len(self.rounds_history) < self.max_rounds:
                        await asyncio.sleep(5)  # Brief pause between rounds
                        await self.start_federated_round(self.model_manager.model_architecture)
                    else:
                        logger.info("Reached maximum number of rounds")
                        await self._finalize_federated_learning()
            else:
                logger.error("Failed to update global model")
                self.current_round.status = FederatedLearningStatus.FAILED
                
        except Exception as e:
            logger.error(f"Failed to aggregate model updates: {e}")
            self.current_round.status = FederatedLearningStatus.FAILED
        
        finally:
            # Reset client statuses
            for client_id in self.current_round.participating_clients:
                self.clients[client_id].status = ClientStatus.CONNECTED
    
    def _verify_update_signature(self, update: ModelUpdate) -> bool:
        """Verify the integrity of model update"""
        # Simplified signature verification
        # In production, use proper cryptographic signatures
        expected_signature = hashlib.sha256(
            f"{update.client_id}{update.timestamp}".encode()
        ).hexdigest()
        
        return update.signature == expected_signature
    
    def _calculate_aggregated_metrics(self, updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated metrics from client updates"""
        if not updates:
            return {}
        
        metrics = {}
        total_data_size = sum(update['data_size'] for update in updates)
        
        # Aggregate metrics weighted by data size
        for update in updates:
            weight = update['data_size'] / total_data_size
            for metric_name, metric_value in update['metrics'].items():
                if metric_name not in metrics:
                    metrics[metric_name] = 0
                metrics[metric_name] += metric_value * weight
        
        return metrics
    
    def _check_convergence(self) -> bool:
        """Check if federated learning has converged"""
        if len(self.rounds_history) < 2:
            return False
        
        # Simple convergence check based on loss improvement
        current_loss = self.current_round.aggregated_metrics.get('loss', float('inf'))
        previous_loss = self.rounds_history[-2].aggregated_metrics.get('loss', float('inf'))
        
        if previous_loss == float('inf'):
            return False
        
        improvement = abs(previous_loss - current_loss) / previous_loss
        return improvement < self.convergence_threshold
    
    async def _finalize_federated_learning(self):
        """Finalize federated learning process"""
        logger.info("Finalizing federated learning process")
        
        # Notify all clients
        for client_id in self.clients:
            await self._notify_client_training_complete(client_id)
        
        # Store final model
        if self.redis_client:
            final_model_data = {
                'model_weights': base64.b64encode(self.model_manager.get_model_weights()).decode(),
                'model_version': self.model_manager.model_version,
                'rounds_completed': len(self.rounds_history),
                'final_metrics': self.current_round.aggregated_metrics if self.current_round else {},
                'completion_time': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.set(
                "federated_final_model",
                json.dumps(final_model_data),
                ex=86400  # Keep for 24 hours
            )
        
        self.current_round = None
    
    async def _notify_client_start_training(self, client_id: str, aggregation_key: str):
        """Notify client to start training"""
        # In production, send actual notification to client
        logger.info(f"Notifying client {client_id} to start training")
        
        notification = {
            'action': 'start_training',
            'round_id': self.current_round.round_id,
            'model_weights': base64.b64encode(self.current_round.global_model_weights).decode(),
            'aggregation_key': aggregation_key,
            'privacy_budget': self.clients[client_id].privacy_budget
        }
        
        # Store notification for client to retrieve
        if self.redis_client:
            await self.redis_client.set(
                f"client_notification:{client_id}",
                json.dumps(notification),
                ex=3600
            )
    
    async def _notify_client_training_complete(self, client_id: str):
        """Notify client that training is complete"""
        logger.info(f"Notifying client {client_id} that training is complete")
        
        notification = {
            'action': 'training_complete',
            'final_model_version': self.model_manager.model_version,
            'rounds_completed': len(self.rounds_history)
        }
        
        if self.redis_client:
            await self.redis_client.set(
                f"client_notification:{client_id}",
                json.dumps(notification),
                ex=3600
            )
    
    async def get_federated_status(self) -> Dict[str, Any]:
        """Get current federated learning status"""
        status = {
            'current_round': asdict(self.current_round) if self.current_round else None,
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c.status != ClientStatus.DISCONNECTED]),
            'rounds_completed': len(self.rounds_history),
            'model_version': self.model_manager.model_version,
            'convergence_threshold': self.convergence_threshold,
            'privacy_budget_remaining': self.privacy_manager.epsilon
        }
        
        return status

# Global federated learning coordinator
_federated_coordinator = None

def get_federated_coordinator(redis_client=None):
    """Get global federated learning coordinator"""
    global _federated_coordinator
    
    if _federated_coordinator is None:
        _federated_coordinator = FederatedLearningCoordinator(redis_client)
    
    return _federated_coordinator

async def initialize_federated_learning():
    """Initialize federated learning system"""
    logger.info("Initializing ScrollIntel federated learning system")
    
    if not HAS_TORCH and not HAS_TENSORFLOW:
        logger.warning("Neither PyTorch nor TensorFlow available. Using simple models only.")
    
    logger.info("Federated learning system initialized successfully")