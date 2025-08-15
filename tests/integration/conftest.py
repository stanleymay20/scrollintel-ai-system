"""
Simple Integration Test Configuration
Provides minimal fixtures for integration testing without full app dependencies
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock

from scrollintel.core.registry import AgentRegistry


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing"""
    # Sample CSV data
    csv_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'Item_{i}' for i in range(1, 101)],
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'date': pd.date_range('2023-01-01', periods=100)
    })
    
    # Sample time series data
    ts_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=365, freq='D'),
        'sales': np.random.randn(365).cumsum() + 1000,
        'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365)
    })
    
    # Sample ML data
    ml_data = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    return {
        'csv_data': csv_data,
        'time_series': ts_data,
        'ml_data': ml_data
    }


@pytest.fixture
def mock_ai_services():
    """Mock external AI services"""
    mocks = {
        'openai': Mock(),
        'anthropic': Mock(),
        'huggingface': Mock(),
        'pinecone': Mock()
    }
    
    # Configure mock responses
    mocks['openai'].chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Mocked OpenAI response"))]
    )
    
    mocks['anthropic'].messages.create.return_value = Mock(
        content=[Mock(text="Mocked Claude response")]
    )
    
    return mocks


@pytest.fixture
def agent_registry():
    """Create test agent registry"""
    registry = AgentRegistry()
    return registry