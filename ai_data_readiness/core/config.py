"""Configuration management for AI Data Readiness Platform."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_data_readiness"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    max_workers: int = 4
    batch_size: int = 1000
    memory_limit_gb: int = 8
    temp_directory: str = "/tmp/ai_data_readiness"
    enable_distributed: bool = False


@dataclass
class QualityConfig:
    """Quality assessment configuration."""
    completeness_threshold: float = 0.95
    accuracy_threshold: float = 0.90
    consistency_threshold: float = 0.85
    validity_threshold: float = 0.90
    ai_readiness_threshold: float = 0.80


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    drift_threshold: float = 0.1
    alert_email: Optional[str] = None
    monitoring_interval_minutes: int = 60
    enable_real_time_monitoring: bool = True


class Config:
    """Main configuration class for AI Data Readiness Platform."""
    
    def __init__(self):
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "ai_data_readiness"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        
        self.processing = ProcessingConfig(
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            batch_size=int(os.getenv("BATCH_SIZE", "1000")),
            memory_limit_gb=int(os.getenv("MEMORY_LIMIT_GB", "8")),
            temp_directory=os.getenv("TEMP_DIR", "/tmp/ai_data_readiness")
        )
        
        self.quality = QualityConfig(
            completeness_threshold=float(os.getenv("COMPLETENESS_THRESHOLD", "0.95")),
            accuracy_threshold=float(os.getenv("ACCURACY_THRESHOLD", "0.90")),
            consistency_threshold=float(os.getenv("CONSISTENCY_THRESHOLD", "0.85")),
            validity_threshold=float(os.getenv("VALIDITY_THRESHOLD", "0.90")),
            ai_readiness_threshold=float(os.getenv("AI_READINESS_THRESHOLD", "0.80"))
        )
        
        self.monitoring = MonitoringConfig(
            drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.1")),
            alert_email=os.getenv("ALERT_EMAIL"),
            monitoring_interval_minutes=int(os.getenv("MONITORING_INTERVAL", "60"))
        )
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Ensure temp directory exists
        Path(self.processing.temp_directory).mkdir(parents=True, exist_ok=True)
        
        # Validate thresholds are between 0 and 1
        thresholds = [
            self.quality.completeness_threshold,
            self.quality.accuracy_threshold,
            self.quality.consistency_threshold,
            self.quality.validity_threshold,
            self.quality.ai_readiness_threshold,
            self.monitoring.drift_threshold
        ]
        
        for threshold in thresholds:
            if not 0 <= threshold <= 1:
                raise ValueError(f"Threshold {threshold} must be between 0 and 1")
        
        return True


# Global configuration instance
_config_instance = None


def get_settings() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance