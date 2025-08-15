"""
Tests for advanced feature engineering capabilities.

This module tests the enhanced transformation pipeline, temporal feature generation,
and dimensionality reduction functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine
from ai_data_readiness.engines.transformation_pipeline import (
    AdvancedTransformationPipeline, PipelineConfig
)
from ai_data_readiness.engines.temporal_feature_generator import (
    AdvancedTemporalFeatureGenerator, TemporalConfig
)
from ai_data_readiness.engines.dimensionality_reduction import (
    Advance