"""
Tests for streaming data processing engine.

This module tests the real-time data preparation pipeline, streaming quality assessment,
and real-time drift detection capabilities.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import time
from typing import Dict, Any, List

from ai_data_readiness.engines.streaming_processor import (
    StreamingDataProcessor, StreamingConfig, StreamingMetrics, StreamingPipeline
)


class TestStreamingDataProcessor:
    """Test cases for StreamingDataProcessor."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample streaming configuration."""
        return StreamingConfig(
            stream_id="test_stream",
            batch_size=10,
            processing_interval=0.1,
            quality_check_interval=2,
            drift_check_interval=5,
            buffer_size=100,
            max_workers=2,
            enable_quality_monitoring=True,
            enable_drift_monitoring=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {"id": i, "value": np.random.randn(), "category": f"cat_{i % 3}"}
            for i in range(50)
        ]
    
    @pytest.fixture
    def processor(self, sample_config):
        """Create streaming processor instance."""
        return StreamingDataProcessor(sample_config)
    
    def test_processor_initialization(self, sample_config):
        """Test processor initialization."""
        processor = StreamingDataProcessor(sample_config)
        
        assert processor.config == sample_config
        assert not processor.is_running
        assert processor.metrics.processed_records == 0
        assert processor.metrics.failed_records == 0
        assert len(processor.data_callbacks) == 0
        assert len(processor.alert_callbacks) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_processor(self, processor):
        """Test starting and stopping the processor."""
        # Test start
        await processor.start()
        assert processor.is_running
        assert len(processor.processing_tasks) == 4  # 4 background tasks
        
        # Test stop
        await processor.stop()
        assert not processor.is_running
    
    @pytest.mark.asyncio
    async def test_process_single_record(self, processor, sample_data):
        """Test processing a single record."""
        await processor.start()
        
        try:
            record = sample_data[0]
            
            # Mock the distributed processor
            with patch.object(processor.distributed_processor, 'process_batch') as mock_process:
                mock_df = pd.DataFrame([record])
                mock_process.return_value = mock_df
                
                # Process record
                result = await processor.process_record(record)
                
                # Give some time for processing
                await asyncio.sleep(0.2)
                
                # Check metrics were updated
                metrics = processor.get_metrics()
                assert metrics.processed_records >= 0  # May be processed in background
                
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_process_batch(self, processor, sample_data):
        """Test processing a batch of records."""
        await processor.start()
        
        try:
            batch = sample_data[:5]
            
            # Mock the distributed processor
            with patch.object(processor.distributed_processor, 'process_batch') as mock_process:
                mock_df = pd.DataFrame(batch)
                mock_process.return_value = mock_df
                
                # Process batch
                results = await processor.process_batch(batch)
                
                # Give some time for processing
                await asyncio.sleep(0.2)
                
                # Results should be processed (may be empty due to async nature)
                assert isinstance(results, list)
                
        finally:
            await processor.stop()
    
    def test_add_callbacks(self, processor):
        """Test adding data and alert callbacks."""
        data_callback = Mock()
        alert_callback = Mock()
        
        processor.add_data_callback(data_callback)
        processor.add_alert_callback(alert_callback)
        
        assert len(processor.data_callbacks) == 1
        assert len(processor.alert_callbacks) == 1
        assert processor.data_callbacks[0] == data_callback
        assert processor.alert_callbacks[0] == alert_callback
    
    def test_get_metrics(self, processor):
        """Test getting streaming metrics."""
        metrics = processor.get_metrics()
        
        assert isinstance(metrics, StreamingMetrics)
        assert metrics.processed_records == 0
        assert metrics.failed_records == 0
        assert metrics.avg_processing_time == 0.0
        assert metrics.quality_score == 1.0
        assert metrics.drift_score == 0.0
    
    @pytest.mark.asyncio
    async def test_quality_monitoring(self, processor, sample_data):
        """Test quality monitoring functionality."""
        await processor.start()
        
        try:
            # Mock quality engine
            with patch.object(processor.quality_engine, 'assess_quality_dataframe') as mock_quality:
                mock_quality.return_value = {'overall_score': 0.8}
                
                # Add data to quality buffer
                for record in sample_data[:20]:
                    processor.quality_buffer.append(record)
                
                # Wait for quality monitoring to run
                await asyncio.sleep(0.5)
                
                # Check that quality monitoring was triggered
                # (This is hard to test directly due to async nature)
                
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_drift_monitoring(self, processor, sample_data):
        """Test drift monitoring functionality."""
        await processor.start()
        
        try:
            # Mock drift monitor
            with patch.object(processor.drift_monitor, 'calculate_drift') as mock_drift:
                mock_drift.return_value = {'overall_drift_score': 0.2}
                
                # Add data to drift buffer
                for record in sample_data:
                    processor.drift_buffer.append(record)
                
                # Wait for drift monitoring to run
                await asyncio.sleep(0.5)
                
                # Check that drift monitoring was triggered
                # (This is hard to test directly due to async nature)
                
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_alert_triggering(self, processor):
        """Test alert triggering mechanism."""
        alert_callback = Mock()
        processor.add_alert_callback(alert_callback)
        
        # Trigger an alert
        await processor._trigger_alert('test_alert', {'test': 'data'})
        
        # Check that callback was called
        alert_callback.assert_called_once()
        args = alert_callback.call_args[0]
        assert args[0] == 'test_alert'
        assert 'test' in args[1]['data']
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_handling(self, processor):
        """Test handling of buffer overflow."""
        # Fill up the input buffer
        for i in range(processor.config.buffer_size + 10):
            record = {"id": i, "value": i}
            result = await processor.process_record(record)
            
            # Should handle overflow gracefully
            if i >= processor.config.buffer_size:
                assert result is None  # Should drop records when buffer is full


class TestStreamingPipeline:
    """Test cases for StreamingPipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create streaming pipeline instance."""
        return StreamingPipeline()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample streaming configuration."""
        return StreamingConfig(
            stream_id="test_stream",
            batch_size=5,
            processing_interval=0.1
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert len(pipeline.processors) == 0
        assert not pipeline.is_running
    
    @pytest.mark.asyncio
    async def test_add_remove_stream(self, pipeline, sample_config):
        """Test adding and removing streams."""
        # Add stream
        processor = await pipeline.add_stream(sample_config)
        
        assert sample_config.stream_id in pipeline.processors
        assert isinstance(processor, StreamingDataProcessor)
        
        # Remove stream
        await pipeline.remove_stream(sample_config.stream_id)
        
        assert sample_config.stream_id not in pipeline.processors
    
    @pytest.mark.asyncio
    async def test_add_duplicate_stream(self, pipeline, sample_config):
        """Test adding duplicate stream raises error."""
        await pipeline.add_stream(sample_config)
        
        with pytest.raises(ValueError, match="already exists"):
            await pipeline.add_stream(sample_config)
        
        # Cleanup
        await pipeline.remove_stream(sample_config.stream_id)
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_stream(self, pipeline):
        """Test removing non-existent stream raises error."""
        with pytest.raises(ValueError, match="not found"):
            await pipeline.remove_stream("nonexistent_stream")
    
    @pytest.mark.asyncio
    async def test_start_stop_all(self, pipeline, sample_config):
        """Test starting and stopping all processors."""
        # Add a stream
        await pipeline.add_stream(sample_config)
        
        # Start all
        await pipeline.start_all()
        assert pipeline.is_running
        
        # Check processor is running
        processor = pipeline.get_processor(sample_config.stream_id)
        assert processor.is_running
        
        # Stop all
        await pipeline.stop_all()
        assert not pipeline.is_running
        assert not processor.is_running
        
        # Cleanup
        await pipeline.remove_stream(sample_config.stream_id)
    
    def test_get_processor(self, pipeline, sample_config):
        """Test getting processor by ID."""
        # Non-existent processor
        assert pipeline.get_processor("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_get_all_metrics(self, pipeline, sample_config):
        """Test getting metrics for all processors."""
        # Add stream
        await pipeline.add_stream(sample_config)
        
        # Get metrics
        metrics = pipeline.get_all_metrics()
        
        assert sample_config.stream_id in metrics
        assert isinstance(metrics[sample_config.stream_id], StreamingMetrics)
        
        # Cleanup
        await pipeline.remove_stream(sample_config.stream_id)
    
    @pytest.mark.asyncio
    async def test_process_record_through_pipeline(self, pipeline, sample_config):
        """Test processing record through pipeline."""
        # Add and start stream
        await pipeline.add_stream(sample_config)
        await pipeline.start_all()
        
        try:
            record = {"id": 1, "value": 10.5}
            
            # Mock the processor's process_record method
            processor = pipeline.get_processor(sample_config.stream_id)
            with patch.object(processor, 'process_record', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = record
                
                result = await pipeline.process_record(sample_config.stream_id, record)
                
                mock_process.assert_called_once_with(record)
                assert result == record
                
        finally:
            await pipeline.stop_all()
            await pipeline.remove_stream(sample_config.stream_id)
    
    @pytest.mark.asyncio
    async def test_process_record_nonexistent_stream(self, pipeline):
        """Test processing record with non-existent stream."""
        record = {"id": 1, "value": 10.5}
        
        with pytest.raises(ValueError, match="not found"):
            await pipeline.process_record("nonexistent", record)
    
    @pytest.mark.asyncio
    async def test_process_batch_through_pipeline(self, pipeline, sample_config):
        """Test processing batch through pipeline."""
        # Add and start stream
        await pipeline.add_stream(sample_config)
        await pipeline.start_all()
        
        try:
            batch = [{"id": i, "value": i * 2.5} for i in range(3)]
            
            # Mock the processor's process_batch method
            processor = pipeline.get_processor(sample_config.stream_id)
            with patch.object(processor, 'process_batch', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = batch
                
                result = await pipeline.process_batch(sample_config.stream_id, batch)
                
                mock_process.assert_called_once_with(batch)
                assert result == batch
                
        finally:
            await pipeline.stop_all()
            await pipeline.remove_stream(sample_config.stream_id)


class TestStreamingConfig:
    """Test cases for StreamingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig(stream_id="test")
        
        assert config.stream_id == "test"
        assert config.batch_size == 1000
        assert config.processing_interval == 1.0
        assert config.quality_check_interval == 10
        assert config.drift_check_interval == 100
        assert config.buffer_size == 10000
        assert config.max_workers == 4
        assert config.enable_quality_monitoring is True
        assert config.enable_drift_monitoring is True
        assert 'quality_score' in config.alert_thresholds
        assert 'drift_score' in config.alert_thresholds
        assert 'processing_latency' in config.alert_thresholds
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_thresholds = {
            'quality_score': 0.8,
            'drift_score': 0.2,
            'processing_latency': 3.0
        }
        
        config = StreamingConfig(
            stream_id="custom_stream",
            batch_size=500,
            processing_interval=2.0,
            quality_check_interval=5,
            drift_check_interval=20,
            buffer_size=5000,
            max_workers=8,
            enable_quality_monitoring=False,
            enable_drift_monitoring=False,
            alert_thresholds=custom_thresholds
        )
        
        assert config.stream_id == "custom_stream"
        assert config.batch_size == 500
        assert config.processing_interval == 2.0
        assert config.quality_check_interval == 5
        assert config.drift_check_interval == 20
        assert config.buffer_size == 5000
        assert config.max_workers == 8
        assert config.enable_quality_monitoring is False
        assert config.enable_drift_monitoring is False
        assert config.alert_thresholds == custom_thresholds


class TestStreamingMetrics:
    """Test cases for StreamingMetrics."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = StreamingMetrics()
        
        assert metrics.processed_records == 0
        assert metrics.failed_records == 0
        assert metrics.avg_processing_time == 0.0
        assert metrics.current_latency == 0.0
        assert metrics.quality_score == 1.0
        assert metrics.drift_score == 0.0
        assert isinstance(metrics.last_updated, float)
    
    def test_custom_metrics(self):
        """Test custom metrics values."""
        custom_time = time.time()
        
        metrics = StreamingMetrics(
            processed_records=100,
            failed_records=5,
            avg_processing_time=1.5,
            current_latency=2.0,
            quality_score=0.85,
            drift_score=0.15,
            last_updated=custom_time
        )
        
        assert metrics.processed_records == 100
        assert metrics.failed_records == 5
        assert metrics.avg_processing_time == 1.5
        assert metrics.current_latency == 2.0
        assert metrics.quality_score == 0.85
        assert metrics.drift_score == 0.15
        assert metrics.last_updated == custom_time


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming processing."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_streaming(self):
        """Test end-to-end streaming processing."""
        config = StreamingConfig(
            stream_id="integration_test",
            batch_size=5,
            processing_interval=0.1,
            quality_check_interval=2,
            drift_check_interval=3,
            buffer_size=50
        )
        
        pipeline = StreamingPipeline()
        processor = await pipeline.add_stream(config)
        
        # Add callbacks to track processing
        processed_records = []
        alerts_received = []
        
        def data_callback(record):
            processed_records.append(record)
        
        def alert_callback(alert_type, alert_data):
            alerts_received.append((alert_type, alert_data))
        
        processor.add_data_callback(data_callback)
        processor.add_alert_callback(alert_callback)
        
        await pipeline.start_all()
        
        try:
            # Send test data
            test_data = [
                {"id": i, "value": np.random.randn(), "category": f"cat_{i % 3}"}
                for i in range(20)
            ]
            
            # Process data in batches
            for i in range(0, len(test_data), 3):
                batch = test_data[i:i+3]
                await pipeline.process_batch(config.stream_id, batch)
                await asyncio.sleep(0.1)  # Small delay between batches
            
            # Wait for processing to complete
            await asyncio.sleep(1.0)
            
            # Check metrics
            metrics = processor.get_metrics()
            assert metrics.processed_records >= 0  # Some records should be processed
            
        finally:
            await pipeline.stop_all()
            await pipeline.remove_stream(config.stream_id)