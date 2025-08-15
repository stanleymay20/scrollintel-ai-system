"""
Demo of Streaming Data Processing Engine for AI Data Readiness Platform

This demo showcases the real-time data preparation pipeline, streaming quality assessment,
and real-time drift detection capabilities.
"""

import asyncio
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified streaming processor for demo
@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""
    stream_id: str
    batch_size: int = 100
    processing_interval: float = 1.0
    quality_check_interval: int = 5
    drift_check_interval: int = 10
    enable_quality_monitoring: bool = True
    enable_drift_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'quality_score': 0.7,
        'drift_score': 0.3
    })

@dataclass
class StreamingMetrics:
    """Metrics for streaming processing performance"""
    processed_records: int = 0
    failed_records: int = 0
    avg_processing_time: float = 0.0
    quality_score: float = 1.0
    drift_score: float = 0.0
    last_updated: float = field(default_factory=time.time)

class StreamingDataProcessor:
    """Simplified streaming data processor for demo"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.metrics = StreamingMetrics()
        self.is_running = False
        self.processed_data = []
        self.quality_buffer = []
        self.drift_buffer = []
        self.alerts = []
        
        logger.info(f"Initialized streaming processor for {config.stream_id}")
    
    async def start(self):
        """Start the streaming processor"""
        self.is_running = True
        logger.info(f"Started streaming processor for {self.config.stream_id}")
    
    async def stop(self):
        """Stop the streaming processor"""
        self.is_running = False
        logger.info(f"Stopped streaming processor for {self.config.stream_id}")
    
    async def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record"""
        start_time = time.time()
        
        try:
            # Simulate data processing
            processed_record = record.copy()
            processed_record['processed_at'] = time.time()
            processed_record['processor_id'] = self.config.stream_id
            
            # Add some processing logic
            if 'value' in record:
                processed_record['normalized_value'] = (record['value'] - 50) / 25
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.processed_records += 1
            self.metrics.avg_processing_time = (
                (self.metrics.avg_processing_time * 0.9) + (processing_time * 0.1)
            )
            self.metrics.last_updated = time.time()
            
            # Store for monitoring
            self.processed_data.append(processed_record)
            self.quality_buffer.append(processed_record)
            self.drift_buffer.append(processed_record)
            
            # Trigger monitoring checks
            await self._check_quality()
            await self._check_drift()
            
            return processed_record
            
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            self.metrics.failed_records += 1
            return record
    
    async def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records"""
        processed_records = []
        
        for record in records:
            processed = await self.process_record(record)
            processed_records.append(processed)
        
        return processed_records
    
    async def _check_quality(self):
        """Check data quality"""
        if not self.config.enable_quality_monitoring:
            return
        
        if len(self.quality_buffer) >= self.config.batch_size:
            # Simulate quality assessment
            df = pd.DataFrame(self.quality_buffer[-self.config.batch_size:])
            
            # Calculate simple quality metrics
            completeness = 1.0 - (df.isnull().sum().sum() / df.size)
            consistency = 1.0 - (df.duplicated().sum() / len(df))
            
            # Simple validity check (no extreme outliers)
            validity = 1.0
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                        validity *= max(0.0, 1.0 - (outliers / len(df)))
            
            quality_score = (completeness + consistency + validity) / 3
            self.metrics.quality_score = quality_score
            
            # Check for quality alerts
            if quality_score < self.config.alert_thresholds['quality_score']:
                alert = {
                    'type': 'quality_degradation',
                    'timestamp': time.time(),
                    'quality_score': quality_score,
                    'threshold': self.config.alert_thresholds['quality_score']
                }
                self.alerts.append(alert)
                logger.warning(f"Quality alert: score {quality_score:.3f} below threshold {self.config.alert_thresholds['quality_score']}")
    
    async def _check_drift(self):
        """Check for data drift"""
        if not self.config.enable_drift_monitoring:
            return
        
        if len(self.drift_buffer) >= self.config.batch_size * 2:
            # Use first half as reference, second half as current
            mid_point = len(self.drift_buffer) // 2
            reference_data = pd.DataFrame(self.drift_buffer[:mid_point])
            current_data = pd.DataFrame(self.drift_buffer[mid_point:])
            
            # Simple drift detection using mean differences
            drift_scores = []
            
            for col in reference_data.select_dtypes(include=[np.number]).columns:
                if col in current_data.columns:
                    ref_mean = reference_data[col].mean()
                    curr_mean = current_data[col].mean()
                    ref_std = reference_data[col].std()
                    
                    if ref_std > 0:
                        # Normalized difference
                        drift_score = abs(curr_mean - ref_mean) / ref_std
                        drift_scores.append(min(1.0, drift_score))
            
            overall_drift = np.mean(drift_scores) if drift_scores else 0.0
            self.metrics.drift_score = overall_drift
            
            # Check for drift alerts
            if overall_drift > self.config.alert_thresholds['drift_score']:
                alert = {
                    'type': 'data_drift_detected',
                    'timestamp': time.time(),
                    'drift_score': overall_drift,
                    'threshold': self.config.alert_thresholds['drift_score']
                }
                self.alerts.append(alert)
                logger.warning(f"Drift alert: score {overall_drift:.3f} above threshold {self.config.alert_thresholds['drift_score']}")
    
    def get_metrics(self) -> StreamingMetrics:
        """Get current metrics"""
        return self.metrics
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts"""
        return self.alerts.copy()

class StreamingPipeline:
    """Pipeline for managing multiple streaming processors"""
    
    def __init__(self):
        self.processors = {}
        self.is_running = False
        logger.info("Initialized streaming pipeline")
    
    async def add_stream(self, config: StreamingConfig) -> StreamingDataProcessor:
        """Add a new streaming processor"""
        if config.stream_id in self.processors:
            raise ValueError(f"Stream {config.stream_id} already exists")
        
        processor = StreamingDataProcessor(config)
        self.processors[config.stream_id] = processor
        
        if self.is_running:
            await processor.start()
        
        logger.info(f"Added stream: {config.stream_id}")
        return processor
    
    async def start_all(self):
        """Start all processors"""
        self.is_running = True
        for processor in self.processors.values():
            await processor.start()
        logger.info("Started all streaming processors")
    
    async def stop_all(self):
        """Stop all processors"""
        self.is_running = False
        for processor in self.processors.values():
            await processor.stop()
        logger.info("Stopped all streaming processors")
    
    def get_processor(self, stream_id: str) -> StreamingDataProcessor:
        """Get processor by ID"""
        return self.processors.get(stream_id)
    
    def get_all_metrics(self) -> Dict[str, StreamingMetrics]:
        """Get metrics for all processors"""
        return {
            stream_id: processor.get_metrics()
            for stream_id, processor in self.processors.items()
        }

async def generate_sample_data(processor: StreamingDataProcessor, num_records: int = 100):
    """Generate sample data for testing"""
    logger.info(f"Generating {num_records} sample records...")
    
    for i in range(num_records):
        # Generate sample record
        record = {
            'id': i,
            'timestamp': time.time(),
            'value': np.random.normal(50, 15),  # Normal distribution
            'category': np.random.choice(['A', 'B', 'C']),
            'sensor_reading': np.random.exponential(2.0),
            'status': np.random.choice(['active', 'inactive'], p=[0.8, 0.2])
        }
        
        # Add some drift after halfway point
        if i > num_records // 2:
            record['value'] += 20  # Introduce drift
            record['sensor_reading'] *= 1.5
        
        # Add some quality issues
        if np.random.random() < 0.1:  # 10% missing values
            record['value'] = None
        
        if np.random.random() < 0.05:  # 5% duplicates
            record = records[-1] if 'records' in locals() and records else record
        
        # Process the record
        await processor.process_record(record)
        
        # Small delay to simulate real-time processing
        await asyncio.sleep(0.01)
    
    logger.info("Sample data generation completed")

async def main():
    """Main demo function"""
    logger.info("=== AI Data Readiness Platform - Streaming Processor Demo ===")
    
    # Create streaming pipeline
    pipeline = StreamingPipeline()
    
    # Configure streaming processor
    config = StreamingConfig(
        stream_id="demo_stream",
        batch_size=20,
        processing_interval=0.5,
        quality_check_interval=2,
        drift_check_interval=3,
        enable_quality_monitoring=True,
        enable_drift_monitoring=True,
        alert_thresholds={
            'quality_score': 0.8,
            'drift_score': 0.4
        }
    )
    
    # Add stream to pipeline
    processor = await pipeline.add_stream(config)
    await pipeline.start_all()
    
    try:
        # Generate and process sample data
        await generate_sample_data(processor, num_records=150)
        
        # Wait a bit for final processing
        await asyncio.sleep(2.0)
        
        # Display results
        logger.info("\n=== Processing Results ===")
        metrics = processor.get_metrics()
        logger.info(f"Processed Records: {metrics.processed_records}")
        logger.info(f"Failed Records: {metrics.failed_records}")
        logger.info(f"Average Processing Time: {metrics.avg_processing_time:.4f}s")
        logger.info(f"Quality Score: {metrics.quality_score:.3f}")
        logger.info(f"Drift Score: {metrics.drift_score:.3f}")
        
        # Display alerts
        alerts = processor.get_alerts()
        logger.info(f"\n=== Alerts Generated ===")
        logger.info(f"Total Alerts: {len(alerts)}")
        
        for alert in alerts[-5:]:  # Show last 5 alerts
            alert_time = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
            logger.info(f"[{alert_time}] {alert['type']}: {alert}")
        
        # Display sample processed data
        logger.info(f"\n=== Sample Processed Data ===")
        sample_data = processor.processed_data[-5:]  # Last 5 records
        for i, record in enumerate(sample_data):
            logger.info(f"Record {i+1}: {record}")
        
        # Performance summary
        logger.info(f"\n=== Performance Summary ===")
        total_time = time.time() - processor.processed_data[0]['processed_at'] if processor.processed_data else 0
        throughput = metrics.processed_records / total_time if total_time > 0 else 0
        logger.info(f"Total Processing Time: {total_time:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} records/second")
        
        # Quality and drift analysis
        logger.info(f"\n=== Quality & Drift Analysis ===")
        logger.info(f"Final Quality Score: {metrics.quality_score:.3f} ({'PASS' if metrics.quality_score >= config.alert_thresholds['quality_score'] else 'FAIL'})")
        logger.info(f"Final Drift Score: {metrics.drift_score:.3f} ({'PASS' if metrics.drift_score <= config.alert_thresholds['drift_score'] else 'FAIL'})")
        
        # Recommendations
        logger.info(f"\n=== Recommendations ===")
        if metrics.quality_score < config.alert_thresholds['quality_score']:
            logger.info("- Quality issues detected. Consider data validation and cleaning.")
        if metrics.drift_score > config.alert_thresholds['drift_score']:
            logger.info("- Data drift detected. Consider model retraining or data source investigation.")
        if len(alerts) > 10:
            logger.info("- High alert volume. Consider adjusting thresholds or improving data sources.")
        
        logger.info("\n=== Demo Completed Successfully ===")
        
    finally:
        # Cleanup
        await pipeline.stop_all()

if __name__ == "__main__":
    asyncio.run(main())