"""
Base Engine class for all ScrollIntel processing engines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class EngineStatus(str, Enum):
    """Status of a processing engine."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class EngineCapability(str, Enum):
    """Capabilities that engines can provide."""
    ML_TRAINING = "ml_training"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    FORECASTING = "forecasting"
    EXPLANATION = "explanation"
    BIAS_DETECTION = "bias_detection"
    FEDERATED_LEARNING = "federated_learning"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    SECURE_STORAGE = "secure_storage"
    REPORT_GENERATION = "report_generation"
    COGNITIVE_REASONING = "cognitive_reasoning"


class BaseEngine(ABC):
    """Abstract base class for all ScrollIntel processing engines."""
    
    def __init__(self, engine_id: str, name: str, capabilities: List[EngineCapability]):
        self.engine_id = engine_id
        self.name = name
        self.capabilities = capabilities
        self.status = EngineStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine with required resources."""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data and return results."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the engine."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the engine."""
        pass
    
    async def health_check(self) -> bool:
        """Perform health check on the engine."""
        try:
            status = self.get_status()
            return status.get("healthy", False)
        except Exception as e:
            logger.error(f"Health check failed for {self.engine_id}: {e}")
            return False
    
    async def start(self) -> None:
        """Start the engine."""
        async with self._lock:
            try:
                await self.initialize()
                self.status = EngineStatus.READY
                logger.info(f"Engine {self.engine_id} started successfully")
            except Exception as e:
                self.status = EngineStatus.ERROR
                self.error_count += 1
                logger.error(f"Failed to start engine {self.engine_id}: {e}")
                raise
    
    async def stop(self) -> None:
        """Stop the engine."""
        async with self._lock:
            try:
                await self.cleanup()
                self.status = EngineStatus.MAINTENANCE
                logger.info(f"Engine {self.engine_id} stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping engine {self.engine_id}: {e}")
                raise
    
    async def execute(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Execute processing with usage tracking."""
        if self.status != EngineStatus.READY:
            raise RuntimeError(f"Engine {self.engine_id} is not ready (status: {self.status})")
        
        async with self._lock:
            try:
                self.status = EngineStatus.PROCESSING
                self.usage_count += 1
                self.last_used = datetime.utcnow()
                
                result = await self.process(input_data, parameters or {})
                
                self.status = EngineStatus.READY
                return result
                
            except Exception as e:
                self.status = EngineStatus.ERROR
                self.error_count += 1
                logger.error(f"Processing failed in engine {self.engine_id}: {e}")
                raise
            finally:
                if self.status == EngineStatus.PROCESSING:
                    self.status = EngineStatus.READY
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.usage_count, 1),
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }