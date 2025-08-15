"""
Streaming Data Processing Engine for AI Data Readiness Platform

This module provides real-time data preparation pipeline, streaming quality assessment,
and real-time drift detection for production AI systems.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import pandas as pd
import numpy as np
from threading import Lock, Event
import queue
import threading
from abc import ABC, abstractmethod

from ai_data_readiness.engines.distributed_processor import DistributedDataProcessor, ProcessingConfig
from ai_data_readiness.engines.drift_monitor import DriftMonitor
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""
    stream_id: str
    batch_size: int = 1000
    processing_interval: float = 1.0  # seconds
    quality_check_interval: int = 10  # batches
    drift_check_interval: int = 100  # batches
    buffer_size: int = 10000
    max_workers: int = 4
    enable_quality_monitoring: bool = True
    enable_drift_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'quality_score': 0.7,
        'drift_score': 0.3,
        'processing_latency': 5.0
    })


@dataclass
class StreamingMetrics:
    """Metrics for streaming processing performance"""
    processed_records: int = 0
    failed_records: int = 0
    avg_processing_time: float = 0.0
    current_latency: float = 0.0
    quality_score: float = 1.0
    drift_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


class StreamingDataProcessor:
    """
    Real-time data processing engine for AI data readiness.
    
    Provides streaming data preparation, quality assessment, and drift monitoring
    capabilities for production AI systems.
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.is_running = False
        self.metrics = StreamingMetrics()
        self._lock = Lock()
        self._stop_event = Event()
        
        # Processing components
        self.distributed_processor = DistributedDataProcessor()
        self.quality_engine = QualityAssessmentEngine()
        self.drift_monitor = DriftMonitor()
        
        # Data buffers
        self.input_buffer = queue.Queue(maxsize=config.buffer_size)
        self.output_buffer = queue.Queue(maxsize=config.buffer_size)
        self.quality_buffer = deque(maxlen=1000)
        self.drift_buffer = deque(maxlen=10000)
        
        # Processing threads
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.processing_tasks = []
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        logger.info(f"Initialized streaming processor for {config.stream_id}")
    
    async def start(self) -> None:
        """Start the streaming data processor"""
        if self.is_running:
            logger.warning("Streaming processor already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._data_processing_loop()),
            asyncio.create_task(self._quality_monitoring_loop()),
            asyncio.create_task(self._drift_monitoring_loop()),
            asyncio.create_task(self._metrics_update_loop())
        ]
        
        logger.info(f"Started streaming processor for {self.config.stream_id}")
    
    async def stop(self) -> None:
        """Stop the streaming data processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info(f"Stopped streaming processor for {self.config.stream_id}")
    
    async def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data record"""
        try:
            # Add to input buffer
            if not self.input_buffer.full():
                self.input_buffer.put(record)
                return await self._get_processed_record()
            else:
                logger.warning("Input buffer full, dropping record")
                return None
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            with self._lock:
                self.metrics.failed_records += 1
            return None
    
    async def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data records"""
        processed_records = []
        
        for record in records:
            processed = await self.process_record(record)
            if processed:
                processed_records.append(processed)
        
        return processed_records
    
    def add_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for processed data"""
        self.data_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics"""
        with self._lock:
            return StreamingMetrics(
                processed_records=self.metrics.processed_records,
                failed_records=self.metrics.failed_records,
                avg_processing_time=self.metrics.avg_processing_time,
                current_latency=self.metrics.current_latency,
                quality_score=self.metrics.quality_score,
                drift_score=self.metrics.drift_score,
                last_updated=self.metrics.last_updated
            )
    
    async def _data_processing_loop(self) -> None:
        """Main data processing loop"""
        batch = []
        last_process_time = time.time()
        
        while self.is_running:
            try:
                # Collect batch
                while len(batch) < self.config.batch_size and not self.input_buffer.empty():
                    try:
                        record = self.input_buffer.get_nowait()
                        batch.append(record)
                    except queue.Empty:
                        break
                
                # Process batch if ready or timeout
                current_time = time.time()
                if (batch and 
                    (len(batch) >= self.config.batch_size or 
                     current_time - last_process_time >= self.config.processing_interval)):
                    
                    start_time = time.time()
                    processed_batch = await self._process_data_batch(batch)
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    with self._lock:
                        self.metrics.processed_records += len(processed_batch)
                        self.metrics.avg_processing_time = (
                            (self.metrics.avg_processing_time * 0.9) + 
                            (processing_time * 0.1)
                        )
                        self.metrics.current_latency = processing_time
                    
                    # Store for quality/drift monitoring
                    if processed_batch:
                        self.quality_buffer.extend(processed_batch)
                        self.drift_buffer.extend(processed_batch)
                    
                    # Notify callbacks
                    for record in processed_batch:
                        for callback in self.data_callbacks:
                            try:
                                callback(record)
                            except Exception as e:
                                logger.error(f"Data callback error: {str(e)}")
                    
                    batch = []
                    last_process_time = current_time
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _process_data_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data records"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(batch)
            
            # Apply transformations using distributed processor
            processing_config = ProcessingConfig(
                batch_size=len(batch),
                max_workers=self.config.max_workers
            )
            
            # Process the data
            processed_df = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.distributed_processor.process_batch,
                df,
                processing_config
            )
            
            # Convert back to records
            return processed_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error processing data batch: {str(e)}")
            with self._lock:
                self.metrics.failed_records += len(batch)
            return []
    
    async def _quality_monitoring_loop(self) -> None:
        """Monitor data quality in real-time"""
        batch_count = 0
        
        while self.is_running:
            try:
                if (self.config.enable_quality_monitoring and 
                    len(self.quality_buffer) >= self.config.batch_size):
                    
                    batch_count += 1
                    
                    if batch_count % self.config.quality_check_interval == 0:
                        # Get recent data for quality assessment
                        recent_data = list(self.quality_buffer)[-self.config.batch_size:]
                        df = pd.DataFrame(recent_data)
                        
                        # Assess quality
                        quality_report = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.quality_engine.assess_quality_dataframe,
                            df
                        )
                        
                        # Update metrics
                        with self._lock:
                            self.metrics.quality_score = quality_report.get('overall_score', 1.0)
                        
                        # Check for quality alerts
                        if self.metrics.quality_score < self.config.alert_thresholds['quality_score']:
                            await self._trigger_alert(
                                'quality_degradation',
                                {
                                    'quality_score': self.metrics.quality_score,
                                    'threshold': self.config.alert_thresholds['quality_score'],
                                    'report': quality_report
                                }
                            )
                
                await asyncio.sleep(self.config.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _drift_monitoring_loop(self) -> None:
        """Monitor data drift in real-time"""
        batch_count = 0
        reference_data = None
        
        while self.is_running:
            try:
                if (self.config.enable_drift_monitoring and 
                    len(self.drift_buffer) >= self.config.batch_size * 2):
                    
                    batch_count += 1
                    
                    if batch_count % self.config.drift_check_interval == 0:
                        # Get reference and current data
                        all_data = list(self.drift_buffer)
                        
                        if reference_data is None:
                            # Use first half as reference
                            mid_point = len(all_data) // 2
                            reference_data = pd.DataFrame(all_data[:mid_point])
                        
                        # Use recent data as current
                        current_data = pd.DataFrame(all_data[-self.config.batch_size:])
                        
                        # Calculate drift
                        drift_report = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.drift_monitor.calculate_drift,
                            current_data,
                            reference_data
                        )
                        
                        # Update metrics
                        with self._lock:
                            self.metrics.drift_score = drift_report.get('overall_drift_score', 0.0)
                        
                        # Check for drift alerts
                        if self.metrics.drift_score > self.config.alert_thresholds['drift_score']:
                            await self._trigger_alert(
                                'data_drift_detected',
                                {
                                    'drift_score': self.metrics.drift_score,
                                    'threshold': self.config.alert_thresholds['drift_score'],
                                    'report': drift_report
                                }
                            )
                
                await asyncio.sleep(self.config.processing_interval * 10)  # Less frequent than quality checks
                
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {str(e)}")
                await asyncio.sleep(10.0)
    
    async def _metrics_update_loop(self) -> None:
        """Update metrics periodically"""
        while self.is_running:
            try:
                with self._lock:
                    self.metrics.last_updated = time.time()
                
                # Check for latency alerts
                if self.metrics.current_latency > self.config.alert_thresholds['processing_latency']:
                    await self._trigger_alert(
                        'high_processing_latency',
                        {
                            'current_latency': self.metrics.current_latency,
                            'threshold': self.config.alert_thresholds['processing_latency']
                        }
                    )
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {str(e)}")
                await asyncio.sleep(30.0)
    
    async def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Trigger an alert"""
        alert_info = {
            'type': alert_type,
            'timestamp': time.time(),
            'stream_id': self.config.stream_id,
            'data': alert_data
        }
        
        logger.warning(f"Alert triggered: {alert_type} for stream {self.config.stream_id}")
        
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_info)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")
    
    async def _get_processed_record(self) -> Optional[Dict[str, Any]]:
        """Get a processed record from output buffer"""
        try:
            return self.output_buffer.get_nowait()
        except queue.Empty:
            return None


class StreamingPipeline:
    """
    High-level streaming pipeline orchestrator.
    
    Manages multiple streaming processors and provides unified interface
    for real-time data preparation and monitoring.
    """
    
    def __init__(self):
        self.processors: Dict[str, StreamingDataProcessor] = {}
        self.is_running = False
        self._lock = Lock()
        
        logger.info("Initialized streaming pipeline")
    
    async def add_stream(self, config: StreamingConfig) -> StreamingDataProcessor:
        """Add a new streaming processor"""
        with self._lock:
            if config.stream_id in self.processors:
                raise ValueError(f"Stream {config.stream_id} already exists")
            
            processor = StreamingDataProcessor(config)
            self.processors[config.stream_id] = processor
            
            if self.is_running:
                await processor.start()
            
            logger.info(f"Added stream processor: {config.stream_id}")
            return processor
    
    async def remove_stream(self, stream_id: str) -> None:
        """Remove a streaming processor"""
        with self._lock:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            await processor.stop()
            del self.processors[stream_id]
            
            logger.info(f"Removed stream processor: {stream_id}")
    
    async def start_all(self) -> None:
        """Start all streaming processors"""
        self.is_running = True
        
        for processor in self.processors.values():
            await processor.start()
        
        logger.info("Started all streaming processors")
    
    async def stop_all(self) -> None:
        """Stop all streaming processors"""
        self.is_running = False
        
        for processor in self.processors.values():
            await processor.stop()
        
        logger.info("Stopped all streaming processors")
    
    def get_processor(self, stream_id: str) -> Optional[StreamingDataProcessor]:
        """Get a streaming processor by ID"""
        return self.processors.get(stream_id)
    
    def get_all_metrics(self) -> Dict[str, StreamingMetrics]:
        """Get metrics for all streaming processors"""
        return {
            stream_id: processor.get_metrics()
            for stream_id, processor in self.processors.items()
        }
    
    async def process_record(self, stream_id: str, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a record through a specific stream"""
        processor = self.processors.get(stream_id)
        if not processor:
            raise ValueError(f"Stream {stream_id} not found")
        
        return await processor.process_record(record)
    
    async def process_batch(self, stream_id: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records through a specific stream"""
        processor = self.processors.get(stream_id)
        if not processor:
            raise ValueError(f"Stream {stream_id} not found")
        
        return await processor.process_batch(records)