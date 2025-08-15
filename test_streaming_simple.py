"""Simple test of streaming processor classes"""

from dataclasses import dataclass, field
from typing import Dict
import time

@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""
    stream_id: str
    batch_size: int = 1000

@dataclass  
class StreamingMetrics:
    """Metrics for streaming processing performance"""
    processed_records: int = 0
    failed_records: int = 0
    last_updated: float = field(default_factory=time.time)

class StreamingDataProcessor:
    """Simple streaming data processor"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.metrics = StreamingMetrics()
    
    def get_metrics(self) -> StreamingMetrics:
        return self.metrics

if __name__ == "__main__":
    config = StreamingConfig("test")
    processor = StreamingDataProcessor(config)
    print("Test successful")
    print("Config:", config)
    print("Metrics:", processor.get_metrics())